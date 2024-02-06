'''s
任务描述，利用RNN对输入的句子进行分词。

相当于对一个序列样本进行标注分类
'''

import torch
import jieba
import torch.nn as nn
import numpy as np
import random
import json
from torch.utils.data import DataLoader


def build_vocab(vocab_path):
    '''
    加载词表
    :param vocab_path:路径
    :return:词表字典
    '''
    vocab={}
    with open(vocab_path,encoding="utf8") as f:
        #从1开始，留出0方便padding
        count=1
        for line in f:
            char=line.strip()
            vocab[char]=count
            count+=1
    vocab['unk'] = len(vocab) + 1
    return vocab

class Dataset:
    def __init__(self,corpus_path,max_length,vocab):
        '''
        构造函数
        :param corpus_path:文本路径
        :param max_length: 最大单词长度
        :param vocab: 字典
        '''
        self.corpus_path=corpus_path
        self.max_length=max_length
        self.vocab=vocab
        self.load()
    def load(self):
        self.data=[]

        with open(self.corpus_path,encoding="utf8") as f:
            for line in f:
                #生产数字序列和对应标签
                senquence=senquence_to_senquence(line,self.vocab)
                label=senquence_to_label(line)
                senquence,label=self.padding(senquence,label)
                senquence,label=torch.LongTensor(senquence),torch.LongTensor(label)
                self.data.append([senquence,label])

                # 使用部分数据做展示，使用全部数据训练时间会相应变长
                if len(self.data) > 10000:
                    break

    def padding(self,sentence,label):
        '''
        将文本补齐或者截断到固定长度
        :param sentence: 句子
        :param label: 标签列表
        :return: padding后的句子
        '''
        senquence=sentence[:self.max_length]
        senquence+=[0]*(self.max_length-len(sentence))
        label=label[:self.max_length]
        label+=[-100]*(self.max_length-len(sentence))
        return senquence,label

    def __len__(self):
        return len(self.data)
    def __getitem__(self, item):
        return self.data[item]

def senquence_to_senquence(sentence,vocab):
    '''
    文本转换为数字序列，为embedding做准备
    :param sentence: 文本
    :param vocab: 字典
    :return: 数字序列->List
    '''
    senquence=[vocab.get(char,vocab['unk']) for char in sentence]

    return senquence

def senquence_to_label(sentence):

    #分词
    words=jieba.lcut(sentence)

    label=[0]*len(sentence)

    #指针指向词的边界
    pointer=0
    for word in words:
        pointer+=len(word)
        label[pointer-1]=1
    return label


#建立数据集
def build_dataset(corpus_path,vocab, max_length, batch_size):
    dataset=Dataset(corpus_path,max_length,vocab)
    dataloader=DataLoader(dataset=dataset,batch_size=batch_size,shuffle=True)
    return dataloader

class Model(nn.Module):
    def __init__(self,input_size,hidden_size,num_rnn_layer,vocab):
        '''
        使用RNN做分词
        :param input_size:输入维度
        :param hidden_size: 隐藏层维度
        :param num_rnn_layer: RNN层数
        :param vocab: 单字表
        '''
        super().__init__()
        #定义Embedding层。同时告诉pad在第一个位置
        self.embedding=nn.Embedding(len(vocab)+1,input_size,padding_idx=0)
        self.rnn=nn.RNN(input_size,hidden_size,batch_first=True,num_layers=num_rnn_layer)
        self.linear=nn.Linear(hidden_size,2)
        self.loss=nn.CrossEntropyLoss(ignore_index=-100)
    def forward(self,x,y=None):
        x=self.embedding(x)
        x,_=self.rnn(x)
        y_pred=self.linear(x)

        if y is not None:
            #计算损失
            y_pred=y_pred.view(-1,2)
            y=y.view(-1)
            loss=self.loss(y_pred,y)
            return loss
        else:
            return y_pred



def main():
    epoch_num = 10  # 训练轮数
    batch_size = 20  # 每次训练样本个数
    char_dim = 50  # 每个字的维度
    hidden_size = 100  # 隐含层维度
    num_rnn_layers = 3  # rnn层数
    max_length = 20  # 样本最大长度
    learning_rate = 1e-3  # 学习率

    vocab_path=r"E:\chars.txt"
    corpus_path=r"E:corpus\corpus.txt"

    vocab=build_vocab(vocab_path)

    dataloader=build_dataset(corpus_path,vocab,max_length,batch_size)

    model=Model(input_size=char_dim,hidden_size=hidden_size,num_rnn_layer=num_rnn_layers,vocab=vocab)
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)  # 建立优化器

    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for x,y in dataloader:
            optim.zero_grad()
            loss=model.forward(x,y)
            loss.backward()
            optim.step()#更新权重
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))

        # 保存模型
    torch.save(model.state_dict(), "rnn_segment_model.pth")
    return

def predict(model_path, vocab_path, input_strings):
    #配置保持和训练时一致
    char_dim = 50  # 每个字的维度
    hidden_size = 100  # 隐含层维度
    num_rnn_layers = 3  # rnn层数
    vocab = build_vocab(vocab_path)       #建立字表
    model = Model(char_dim, hidden_size, num_rnn_layers, vocab)   #建立模型
    model.load_state_dict(torch.load(model_path))   #加载训练好的模型权重
    model.eval()
    for input_string in input_strings:
        #逐条预测
        x = senquence_to_senquence(input_string, vocab)
        with torch.no_grad():
            result = model.forward(torch.LongTensor([x]))[0]
            result = torch.argmax(result, dim=-1)  #预测出的01序列
            #在预测为1的地方切分，将切分后文本打印出来
            for index, p in enumerate(result):
                if p == 1:
                    print(input_string[index], end=" ")
                else:
                    print(input_string[index], end="")
            print()

if __name__ == '__main__':
    main()
    input_strings = ["同时国内有望出台新汽车刺激方案",
                     "沪胶后市有望延续强势",
                     "经过两个交易日的强势调整后",
                     "昨日上海天然橡胶期货价格再度大幅上扬"]
    predict("rnn_segment_model.pth", r"E:\chars.txt", input_strings)
