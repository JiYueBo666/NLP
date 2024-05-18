import jieba
import pandas as pd
from main import load_data
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

cn_stopwords = pd.read_csv('https://mirror.coggle.club/stopwords/baidu_stopwords.txt', header=None)[0].values


train_data,test_data=load_data()

tfidf=TfidfVectorizer(
    tokenizer=jieba.lcut,
    stop_words=list(cn_stopwords),
    #ngram_range=(1,2),
)

from sklearn.model_selection import cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report

train_tfidf=tfidf.fit_transform(train_data['text'])


test_tfidf=tfidf.transform(test_data[0])
cv_pred=cross_val_predict(KNeighborsClassifier(),train_tfidf,train_data['label'],cv=5)
print(classification_report(train_data['label'],cv_pred))

print('-----svc')
model=LinearSVC()

model.fit(train_tfidf,train_data['label'])

cv_pred2=cross_val_predict(model,train_tfidf,train_data['label'],cv=5)
print(classification_report(train_data['label'],cv_pred2))

pd.DataFrame({
 'ID':range(1, len(test_data) + 1),
    "Target":model.predict(test_tfidf)
}).to_csv('svm.csv', index=None)


