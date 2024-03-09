import pandas as pd
import json
from collections import  defaultdict
import numpy as np
import torch
from sklearn.utils import shuffle
from torch.utils.data import DataLoader
from Config import Config
from sklearn.model_selection import train_test_split
from transformers import BertModel,BertTokenizer


class Make_data():
    def __init__(self,config):
        self.split_size=0.7
        self.config=config
        self.cls_sentences=None
        self.labels=None
        self.test_labels=None
        self.mode = None
        self.sentences_encode=None
        self.test_sentences_encode=None
        self.tokenizer = BertTokenizer(config['vocab_path'])
        self.vocab=load_vocab(Config['vocab_path'])
        self.load(Config['Sentence_classfy_path'])


    def load(self,path):
        data=pd.read_csv(path)
        data=shuffle(data)
        data_num=len(data)
        train_data=data[:int(data_num*self.split_size)]
        test_data=data[int(data_num*self.split_size):]

        #得到的sentences是一个列表，存储string类型句子
        sentences,self.labels=list(train_data['review'].values),train_data['label'].values
        #编码为向量
        self.sentences_encode=self.tokenizer.batch_encode_plus(sentences, padding='max_length',max_length=30,truncation=True)
        # self.cls_sentences=sentences_['input_ids']
        #keys: input_ids,attention_mask,token_type_ids

        test_sentences,self.test_labels=list(test_data['review'].values),test_data['label'].values
        self.test_sentences_encode=self.tokenizer.batch_encode_plus(test_sentences, padding='max_length',max_length=30,truncation=True)


    def split_data(self,cls_sentences,labels):
        x_train,x_test,y_train,y_test=train_test_split(cls_sentences,labels,test_size=0.2,random_state=42)
        return x_train,x_test,y_train,y_test

    def __getitem__(self, item):
        res=None
        if self.mode!="test":
            res=(torch.LongTensor(self.sentences_encode['input_ids'][item]),
                 torch.LongTensor(self.sentences_encode['attention_mask'][item]),
                 torch.LongTensor(self.sentences_encode['token_type_ids'][item]),
                 torch.LongTensor([self.labels[item]]))
        else:

            res=(torch.LongTensor(self.test_sentences_encode['input_ids'][item]),
                 torch.LongTensor(self.test_sentences_encode['attention_mask'][item]),
                 torch.LongTensor(self.test_sentences_encode['token_type_ids'][item]),
                 torch.LongTensor([self.test_labels[item]]))


        return res


    def __len__(self):
        if self.mode!="test":
            return len(self.labels)
        else:
            return len(self.test_labels)

def load_vocab(vocab_path):
    vocab=defaultdict(int)
    with open(vocab_path,'r',encoding='utf-8') as f:
        for line in f:
            line=line.strip()
            vocab[line]=vocab.get(line,len(vocab))
    return vocab



if __name__ == '__main__':
    dataset=Make_data(Config)
    dataloader=DataLoader(dataset=dataset,batch_size=Config['batch_size'],shuffle=True,drop_last=True)
    for idx,bat in enumerate(dataloader):
        count=1
        intput_ids,mask,type_id,label=bat



        if count==1:
            break
