import torch
import torch.nn as nn
import numpy
from collections import  defaultdict
import math

class newWordDetect:
    def __init__(self,corpus_path):
        self.max_word_length=5
        self.path=corpus_path
        self.word_count = defaultdict(int)#
        self.left_neighbor = defaultdict(dict)#左边相邻单词
        self.right_neighbor = defaultdict(dict)#右边相邻单词
        self.load_corpus(self.path)
        self.calc_pmi()#凝固度，词的内部应该是稳定的，数学上表示为：log（词出现的概率÷词中每个字出现的概率）
        self.calc_entropy()#左右熵，词的外部应该是多变的。熵比较大
        self.calc_word_values()
    def load_corpus(self,path):
        #加载语料
        with open(path,encoding="utf8") as f:
            for line in f:
                sentence=line.strip()
                for word_length in range(1,self.max_word_length):
                    self.ngram_count(sentence, word_length)
        return
    def ngram_count(self,sentence,word_length):
        '''
        按照窗口长度取词，并记录左右邻居
        :return:
        '''
        for i in range(len(sentence)-word_length+1):

            #给定一个词长度，从按照滑动窗口的方式将窗口内的词加入字典统计
            word=sentence[i:word_length]

            #统计该词的出现次数
            self.word_count[word]+=1

            #某个字在某个词左边的出现次数
            if i-1>0:
                char=sentence[i-1]
                self.left_neighbor[word][char] = self.left_neighbor[word].get(char, 0) + 1
            #同上，这是右边的字
            if i+word_length<len(sentence):
                char=sentence[i+word_length]
                self.right_neighbor[word][char]=self.right_neighbor[word].get(char,0)+1
        return
    #计算熵
    def calc_entropy_by_word_count_dict(self, word_count_dict):

        #计算总字数
        total = sum(word_count_dict.values())

        #计算熵即 -plog(p)
        entropy = sum([-(c / total) * math.log((c / total), 10) for c in word_count_dict.values()])
        return entropy

    #计算左右熵
    def calc_entropy(self):
        self.word_left_entropy = {}
        self.word_right_entropy = {}
        for word, count_dict in self.left_neighbor.items():
            #word为left_neighbor的键值，也就是词
            #count_dict为一个字典，里面存储了所有出现在word左边的字的频率
            self.word_left_entropy[word] = self.calc_entropy_by_word_count_dict(count_dict)
        for word, count_dict in self.right_neighbor.items():
            self.word_right_entropy[word] = self.calc_entropy_by_word_count_dict(count_dict)

    def calc_total_count_by_length(self):
        '''
        统计每种词长度下的词总数
        :return:
        '''
        self.word_count_by_length=defaultdict(int)
        for word,count in self.word_count.items():
            #word和count分别为词和词出现的次数
            self.word_count_by_length[len(word)]+=count
        return
    def calc_pmi(self):
        '''
        计算互信息
        :return:
        '''
        #先统计词总数
        self.calc_total_count_by_length()
        self.pmi={}
        for word,count in self.word_count.items():

            #该词出现的次数/与该词等长的词总数
            p_word=count/self.word_count_by_length[len(word)]

            p_chars=1

            for char in word:
                p_chars *= self.word_count[char] / self.word_count_by_length[1]
                self.pmi[word] = math.log(p_word / p_chars, 10) / len(word)
        return