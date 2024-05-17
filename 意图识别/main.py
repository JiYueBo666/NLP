import pandas as pd
import matplotlib.pyplot as plt

data_dir = 'https://mirror.coggle.club/dataset/coggle-competition/'
train_data = pd.read_csv(data_dir + 'intent-classify/train.csv', sep='\t', header=None)
test_data = pd.read_csv(data_dir + 'intent-classify/test.csv', sep='\t', header=None)

train_data.columns=['text','label']
#print(train_data)

#计算每一行的句子长度。
train_sentence=train_data['text'].apply(len)
train_data['sentence_length']=train_sentence
# plt.hist(train_sentence,bins=20,edgecolor='black')
# plt.title('Sentence Length Distribution')
# plt.xlabel('Length of Sentence')
# plt.ylabel('Frequency')
# plt.show()

#查看不同类的句子长度分布
train_sentence_group=train_data.groupby('label')['sentence_length']
mean_length=train_sentence_group.mean()
median_length=train_sentence_group.median()
print(mean_length)

plt.barh(mean_length.index,mean_length,label='Mean')
plt.show()
# plt.figure(figsize=(10, 6))
# plt.boxplot([group['sentence_length'] for _, group in train_sentence_group], labels=train_data['label'].unique())
# plt.title('Sentence Length Distribution by Label (Box Plot)')
# plt.xlabel('Label')
# plt.ylabel('Length of Sentence')
# plt.xticks(rotation=60)
# plt.show()

char_sum=sum(train_data['sentence_length'])
print("总计字符数",char_sum)