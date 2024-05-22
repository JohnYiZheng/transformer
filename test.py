import torch
import torch.nn as nn
import spacy
import argparse
from data import *
from models.model.transformer import Transformer

def load_model(model_path, device):
    model = Transformer(src_pad_idx=src_pad_idx,
                    trg_pad_idx=trg_pad_idx,
                    trg_sos_idx=trg_sos_idx,
                    d_model=d_model,
                    enc_voc_size=enc_voc_size,
                    dec_voc_size=dec_voc_size,
                    max_len=max_len,
                    ffn_hidden=ffn_hidden,
                    n_head=n_heads,
                    n_layers=n_layers,
                    drop_prob=drop_prob,
                    device=device).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def translate_sentence(model, sentence, src_field, trg_field, device, max_len=50):
    model.eval()
    nlp = spacy.load('en_core_web_sm')
    tokens = [token.text.lower() for token in nlp(sentence)]
    tokens = [src_field.init_token] + tokens + [src_field.eos_token]
    src_indexes = [src_field.vocab.stoi[token] for token in tokens]
    src_tensor = torch.LongTensor(src_indexes).unsqueeze(0).to(device)
    src_mask = model.make_src_mask(src_tensor)
    
    with torch.no_grad():
        encoder_outputs = model.encoder(src_tensor, src_mask)

    trg_indexes = [trg_field.vocab.stoi[trg_field.init_token]]

    for i in range(max_len):
        trg_tensor = torch.LongTensor(trg_indexes).unsqueeze(0).to(device)
        trg_mask = model.make_trg_mask(trg_tensor)
        with torch.no_grad():
            output = model.decoder(trg_tensor, encoder_outputs, trg_mask, src_mask)

        pred_token = output.argmax(2)[:, -1].item()
        trg_indexes.append(pred_token)

        if pred_token == trg_field.vocab.stoi[trg_field.eos_token]:
            break

    trg_tokens = [trg_field.vocab.itos[i] for i in trg_indexes]

    return trg_tokens[1:-1]

def main():
    parser = argparse.ArgumentParser(description='Translate a sentence using a trained Transformer model.')
    parser.add_argument('--model-path', type=str, required=True, help='Path to the trained model weights')

    args = parser.parse_args()

    # 加载字段词汇表
    src_field = loader.source
    trg_field = loader.target

    # 加载模型
    model = load_model(args.model_path, device)

    # 翻译输入句子
    print("Model loaded. You can now input sentences to translate (type 'quit' to exit):")

    while True:
        sentence = input("Enter sentence: ")
        if sentence.lower() == 'quit':
            break

        translation = translate_sentence(model, sentence, src_field, trg_field, device)
        translation_sentence = ' '.join(translation)

        print(f'Translated Sentence: {translation_sentence}')

if __name__ == '__main__':
    main()
