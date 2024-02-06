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

def cut_method(string,word_dict,max_len):
    words=[]

    #开始缩小窗口，直到匹配成功或者窗口大小为0
    while string!="":
        #窗口初始化为当前字符串的最大长度或者预设定的切分的最大长度
        lens=min(max_len,len(string))
        word=string[:lens]
        while word not in word_dict:
            if len(word)==1:
                #如果窗口已经为1了，不管是否匹配必须切分
                break
            word = word[:len(word) - 1]
        words.append(word)
        string=string[len(word):]
    return words





if __name__ == '__main__':
    txt_path=r"E:中文分词\dict.txt"
    word_dict,max_len=load_word_dict(txt_path)

    string="测试字符串"

    cut_result=cut_method(string,word_dict,max_len)
    print(cut_result)
