import torch
import numpy as np
import re
import time

def load_word_dict(path):
    '''
    加载字典
    :param path:文件路径
    :return: 字典
    '''
    word_dict={}#list比较慢
    max_word_length=0
    with open(path,encoding="utf8") as f:
        for line in f:
            word=line.split()[0]#切分的第一部分是实体
            word_dict[word]=0
            max_word_length=max(max_word_length,len(word))
    return word_dict,max_word_length

def cut_method_reverse(word_dict,max_len,string):
    words=[]

    while string!="":
        lens=min(max_len,len(string))
        word=string[-lens:]

        while word not in word_dict:
            if len(word)==1:
                break
            word=word[-len(word)+1:]
        words.append(word)
        string=string[:-len(word)]
    words.reverse()
    return words
if __name__ == '__main__':
    txt_path = r"E:\八斗课程-精品班\文件\week4 中文分词和tfidf特征应用\week4 中文分词和tfidf特征应用\上午-中文分词\dict.txt"
    word_dict, max_len = load_word_dict(txt_path)

    string = "姬越博是谁"

    cut_result = cut_method_reverse(word_dict,max_len,string)
    print(cut_result)







