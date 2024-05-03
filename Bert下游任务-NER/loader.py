import torch
from Config import config
from collections import defaultdict
from torch.utils.data import DataLoader
import jieba
from utils import build_dict

Label={
    'O':0,
    'B-LOC':1,
    'I-LOC':2,
    'B-PER':3,
    'I-PER':4,
    'B-ORG':5,
    'I-ORG':6
}

class Dataset:
    def __init__(self,Config):
        self.config=Config

        self.do_train=True
        self.train_data=self.load_data(self.config['train_path'])
        self.valid_data=self.load_data(self.config['valid_path'])
        self.table=build_dict(self.train_data+self.valid_data)

    def encode_sentence(self,sentence,max_length=config['max_len']):
        result=[]
        if len(sentence)>=max_length:
            sentence=sentence[:max_length]
            result=[self.table[word]for word in sentence]


        elif len(sentence)<max_length:
            result=[self.table[word]for word in sentence]
            result.extend([0]*(max_length-len(sentence)))
        return  result

    @staticmethod
    def load_data(filename):
        D = []
        with open(filename, encoding='utf-8') as f:
            f = f.read()
            for l in f.split('\n\n'):
                if not l:
                    continue
                d = ['']
                label=[]#每个字对应的label
                for i, c in enumerate(l.split('\n')):
                    char, flag = c.split(' ')
                    d[0] += char
                    label.append(flag)
                D.append([d,label])
        return D


    def __getitem__(self, item):
        if self.do_train:
            Encode_text, ps,pe,label=self.TokenizeAndEncode(self.train_data[item])
            return self.train_data[item][0][0],torch.LongTensor(Encode_text), torch.LongTensor(ps), torch.LongTensor(
                pe), torch.LongTensor(label)


        else:
            Encode_text, ps,pe,label = self.TokenizeAndEncode(self.valid_data[item])


            return self.valid_data[item][0][0],torch.LongTensor(Encode_text), torch.LongTensor(ps), torch.LongTensor(pe),torch.LongTensor(label)


    def __len__(self):
        if self.do_train:
            return len(self.train_data)
        else:
            return len(self.valid_data)

    def TokenizeAndEncode(self,data:list):
        '''
        :param data:[[sentence],[label]]
        :return:
        '''

        #第一步，对句子进行分词
        sentence=data[0][0]#str的句子
        label=data[1].copy()#句子的标签。
        #第二步：添加position信息。
        sentence_start_position=[i for i in range(len(sentence))]#对于单个字，tail和head的位置信息相同。
        sentence_end_position=[i for i in range(len(sentence))]

        sentence_cut=jieba.lcut(sentence)#[word1,word2,word3...]

        for word in sentence_cut:
            #确定该单词在原本句子中的位置
            if len(word)>1:
                start=sentence.find(word)
                end=start+len(word)-1

                sentence+=word
                sentence_start_position.extend([start]*len(word))
                sentence_end_position.extend([end]*len(word))

                label.extend([-1]*len(word))


        if len(sentence_start_position)>config['max_len']:
            sentence_start_position=sentence_start_position[:config['max_len']]
            sentence_end_position = sentence_end_position[:config['max_len']]
        else:
            sentence_start_position.extend([-1] * (config['max_len'] - len(sentence_start_position)))
            sentence_end_position.extend([-1] * (config['max_len'] - len(sentence_end_position)))


        #对原句子利用bert进行编码
        encode_text=self.encode_sentence(sentence)

        if len(label)>config['max_len']:
            label=label[:config['max_len']]
        else:
            #对扩展的部分补充标签
            label.extend([-1]*(config['max_len']-len(label)))

        for i in range(len(label)):
            label[i]=Label.get(label[i],-1)
        return encode_text,sentence_start_position,sentence_end_position,label

    def encode(self,sentence):
        sentence_start_position = [i for i in range(len(sentence))]  # 对于单个字，tail和head的位置信息相同。
        sentence_end_position = [i for i in range(len(sentence))]

        sentence_cut = jieba.lcut(sentence)  # [word1,word2,word3...]
        for word in sentence_cut:
            # 确定该单词在原本句子中的位置
            if len(word) > 1:
                start = sentence.find(word)
                end = start + len(word) - 1

                sentence += word
                sentence_start_position.extend([start] * len(word))
                sentence_end_position.extend([end] * len(word))

        if len(sentence_start_position) > config['max_len']:
            sentence_start_position = sentence_start_position[:config['max_len']]
            sentence_end_position = sentence_end_position[:config['max_len']]
        else:
            sentence_start_position.extend([-1] * (config['max_len'] - len(sentence_start_position)))
            sentence_end_position.extend([-1] * (config['max_len'] - len(sentence_end_position)))

        # 对原句子利用bert进行编码
        encode_text = self.encode_sentence(sentence)

        return encode_text, sentence_start_position, sentence_end_position






if __name__ == '__main__':
    data=Dataset(config)
    data.do_train=False



    dataloader=DataLoader(dataset=data,batch_size=config['batch_size'],shuffle=True)

    for i,b in enumerate(dataloader):
        sentence,_,_,_,_=b
        print(len(sentence))