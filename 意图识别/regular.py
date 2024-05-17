import pandas
import re

import pandas as pd
import wordcloud
import matplotlib.pyplot as plt
import jieba
import jieba.analyse
from main import load_data
import re

train_data, test_data = load_data()

def extract_keywords(text,topK=5):
    keywords=jieba.analyse.extract_tags(text,topK=topK)
    return keywords

for category in train_data['label'].unique():
    #获取属于当前类别的文本
    category_texts=train_data[train_data['label']==category]['text'].astype(str)
    #合并所有文档为一个字符串
    combinded_text=''.join(category_texts)
    keywords=extract_keywords(combinded_text)


class regular_text_classfy():
    def __init__(self):
        self.keywords=[
            [['有票', '机票', '飞机', '路线','火车','高铁','地铁','导航'],'Travel-Query'],
            [[ '一首', '歌曲', '随机', '专辑','音乐','歌声','好听'],'Music-Play'],
            [['电影', '主演','影视','主角','戏'],'FilmTele-Play'],
            [['动漫', '花絮','动画'],'Video-Play'],
            [['收听', '频率', '广播电台', '电台'],'Radio-Listen'],
            [['空调', '打开', '模式', '风速', '温度','智能','帮我','给我','机器人'],'HomeAppliance-Control'],
            [['明天', '下雨', '紫外线', '空气质量','打雷','天气','气温'],' Weather-Query'],
            [['闹钟', '提醒',  '明天', '下午'],'Alarm - Update'],
            [['农历', '几号', '几月', '星期', '哪天'],'Calendar - Query'],
             [['播放', '节目', '回放', '播出', '卫视'],'TVProgram - Play'],
            [['播放', '小说', '一下', '故事', '广播剧'],'Audio - Play'],
            #[['一首', '什么', '有声', '歌曲', '漫画'],'Other'],
        ]
    def predict(self,text):
        for item in self.keywords:
            keywords_list,category =item
            for keywords in keywords_list:
                if keywords in text:
                    return category
        return 'Other'

    def test(self,test_data):
        category=[]
        for sentence in test_data[0]:
            cat=self.predict(sentence)
            category.append(cat)
        test_data['label']=pd.DataFrame(category)
        return test_data



if __name__ == '__main__':
    model=regular_text_classfy()
    test_data=model.test(test_data)
    print(test_data)