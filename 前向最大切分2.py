import  json

def load_word_dict(path):
    word_list={}
    with open(path,encoding="utf8") as f:
        for line in f:

            #第一个元素为存储的词
            word=line.split()[0]

            #存储词的所有前缀。
            #注意一点，如果词的某个前缀本身也是一个词，那么把其标记为词而不是标记为前缀
            for i in range(len(word)):
                if word[:i] not in word_list:
                    word_list[word[:i]]=0
            word_list[word]=1
    return word_list


def cut_method(string,word_dict):
    if string=="":
        return []
    #预设定窗口大小
    start_idx,end_idx=0,1

    #存放切分好的词
    word_list=[]

    #从第一个字开始查找
    window=string[start_idx:end_idx]
    find_word=string[start_idx:end_idx]


    #接下来分三种情况：窗口内容不在字典，窗口内容是词，窗口内容是前缀

    #第一种,窗口内容不在词典中，直接切分
    while start_idx<len(string):
        if window not in word_dict or end_idx>len(string):
            #记录找到的词
            word_list.append(find_word)

            #更新起点位置
            start_idx+=len(find_word)
            end_idx=start_idx+1

            #从新位置开始找，重新初始化find_word
            window=string[start_idx:end_idx]
            find_word=window


          #如果窗口内容是一个词，看看能否扩展到比他更长的词
        elif word_dict[window]==1:

            #先记录已经找到了一个词，等一下如果无法扩展，就保存这个词
            find_word=window

            #进行扩展
            end_idx+=1
            window=string[start_idx:end_idx]
        elif word_dict[window]==0:
            #如果是前缀，直接扩展一个字的长度
            end_idx+=1
            window=string[start_idx:end_idx]

    # if window not in word_dict:
    #     word_list += list(window)
    # else:
    #     word_list.append(window)
    return word_list




if __name__ == '__main__':
    txt_path = r"E:\dict.txt"
    string="人工智能迅速发展"
    word_dict=load_word_dict(txt_path)
    res=cut_method(string,word_dict)
    print(res)
