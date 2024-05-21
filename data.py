"""
@author : Hyunwoong
@when : 2019-10-29
@homepage : https://github.com/gusdnd852
"""
from conf import *
from util.data_loader import DataLoader
from util.tokenizer import Tokenizer

tokenizer = Tokenizer()
loader = DataLoader(ext=('.en', '.de'),
                    tokenize_en=tokenizer.tokenize_en,
                    tokenize_de=tokenizer.tokenize_de,
                    init_token='<sos>',
                    eos_token='<eos>')

train, valid, test = loader.make_dataset()
loader.build_vocab(train_data=train, min_freq=2)
train_iter, valid_iter, test_iter = loader.make_iter(train, valid, test,
                                                     batch_size=batch_size,
                                                     device=device)

example_idx = 0  # 假设我们取第一个样本
src_sentence = vars(train.examples[example_idx])['src']
trg_sentence = vars(train.examples[example_idx])['trg']
print("example of train dataset:")
print("src:", src_sentence)
print("trg:", trg_sentence)

src_pad_idx = loader.source.vocab.stoi['<pad>']
trg_pad_idx = loader.target.vocab.stoi['<pad>']
trg_sos_idx = loader.target.vocab.stoi['<sos>']
trg_eos_idx = loader.target.vocab.stoi['<eos>']
print("source pad:", src_pad_idx)
print("target pad:", trg_pad_idx)
print("target sos:", trg_sos_idx)
print("target eos:", trg_eos_idx)

enc_voc_size = len(loader.source.vocab)
dec_voc_size = len(loader.target.vocab)
