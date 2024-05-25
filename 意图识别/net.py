import pandas as pd
import torch
from peft import LoraConfig,TaskType,get_peft_model
from main import load_data
import torch.nn as nn
from torch.nn import LSTM
from transformers import BertModel,BertTokenizer,BertForSequenceClassification
from torch.utils.data import Dataset,DataLoader
from collections import defaultdict
from collections import Counter
from utils import FocalLoss
import numpy as np
from sklearn.metrics import accuracy_score,f1_score,precision_score,recall_score,classification_report
from sklearn.metrics import confusion_matrix as  cmx


train_data,test_data=load_data()
targets=train_data['label'].unique()


def num2target(target_dict):
    num2t={}
    for key,value in target_dict.items():
        num2t[value]=key
    return num2t

def apply_targets(train_data,targets):
    targets_dict=defaultdict(int)
    for i in range(len(targets)):
        targets_dict[targets[i]]=i
    train_data['targets']=train_data['label'].map(targets_dict)

    num2targts=num2target(targets_dict)

    return train_data,num2targts

class DataStore(Dataset):
    def  __init__(self,tokenizer,data:pd.DataFrame,mode):
        super().__init__()
        self.sentence_data=[]
        self.train_label=[]
        self.binary_label=[]
        self.tokenizer=tokenizer
        self.train_length=len(data)
        self.mode=mode
        self.data=data
        self.max_length=40
        self.load()
    def load(self):
        texts = list(self.data['text'])
        self.sentence_data=texts

        self.train_label=list(self.data['targets'])

        for i in range(len(self.train_label)):
            if self.train_label[i]<11:
                self.binary_label.append(1)
            else:
                self.binary_label.append(0)

    def __len__(self):
        return self.train_length

    def __getitem__(self, item):
        encode=self.encode(item)

        return encode,self.train_label[item],self.binary_label[item]


    def encode(self,item):
        text=self.sentence_data[item]
        encode=self.tokenizer.encode_plus(text,max_length=self.max_length,padding='max_length',truncation=True)
        encode={key:torch.tensor(value) for key,value in encode.items()}
        return encode

class Bert_model(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert=BertForSequenceClassification.from_pretrained(r"E:\python\pre_train_model\bert_file",num_labels=len(targets))
        self.softmax=nn.Softmax(dim=-1)
        self.focalloss=FocalLoss()
    def forward(self,inputs,label=None):
        output=self.bert(**inputs).logits
       # output=self.fc(self.dropout(output))
        if label is not None:
            return self.focalloss(output.view(-1,len(targets)),label)
        else:
            return torch.argmax(self.softmax(output),dim=1)

class LSTM_model(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding=nn.Embedding(21129,768)
        self.lstm=nn.LSTM(input_size=768,hidden_size=768,num_layers=2,batch_first=True,bidirectional=True)
        self.linear=nn.Linear(768,768)
        self.dropout=nn.Dropout(0.1)
        self.avgpool=nn.AvgPool1d(kernel_size=40)
        self.maxpool=nn.MaxPool1d(kernel_size=40)

        self.softmax = nn.Softmax(dim=-1)
        self.fc=nn.Linear(768*2,len(targets))

        self.focal_loss=FocalLoss()
    def forward(self,input,label=None):

        #[b,l,h]
        output=self.embedding(input)
        output=self.linear(output)


        #[b,h],output:[batch,length,2*hidden]
        output,_=self.lstm(output)#( input,batch,length,2*hidden),h,c
        output=output[:,-1,:].squeeze()

        output=self.fc(output)

        if label is not None:
            loss=self.focal_loss(output,label)
            return loss
        else:
            return torch.argmax(self.softmax(output),dim=-1)

class fast_text(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(21129, 768)
        self.linear=nn.Linear(768,768)
        self.pool=nn.AvgPool1d(kernel_size=40)
        self.fc=nn.Linear(768,len(targets))
        self.lossfunc=nn.CrossEntropyLoss(weight=torch.FloatTensor([1,1,1,1,1,1,1,1,1,6,6,6]))#weight=torch.FloatTensor([1,1,1,1,1,1,1,1,1,6,6,6])
        self.softmax=nn.Softmax(dim=-1)

        self.focalloss=FocalLoss()

    def forward(self,input,label=None):
        output=self.embedding(input)
      #  output=self.linear(output)
        output=self.pool(output.transpose(-1,-2)).transpose(-1,-2).squeeze()
        output=self.fc(output)

        if label is not None:
        #    return self.lossfunc(output.view(-1,len(targets)),label)
            loss=self.focalloss(output.view(-1,len(targets)),label)
            return loss


        else:
            return torch.argmax(self.softmax(output),dim=-1)


class model_use():
    def __init__(self):
        return

    def fast_text_baseline(self,test_dataloader):

        fasttext=torch.load('fasttext.pth')

        model_correct=[0,0,0]
        vote_correct=0
        wrong_record=[]
        false_classfy=[]

        for batch_idx, (inputs, label,binary_label) in enumerate(test_dataloader):
            for key in inputs.keys():
                inputs[key] = inputs[key].cuda()
            pre=[]
            model = fasttext
            model=model.cuda()
            model.eval()
            pred=model(inputs['input_ids']).squeeze()
            pred_num=pred.detach().cpu().numpy()
            correct=(pred_num==label.cpu().numpy()).sum()
            model_correct[0]+=correct

            batch_wrong=[]
            for i in range(len(pred_num)):
                if pred_num[i]!=label.cpu().numpy()[i]:
                    wrong_record.append(label.cpu().numpy()[i])
                    false_classfy.append(pred_num[i])

            pre.append(pred_num)

            vote_pre=np.array(pre)
            result=[Counter(vote_pre[:,i]).most_common(1)[0][0] for i in range(len(label))]
            vote_correct+=(result==label.detach().cpu().numpy()).sum()
            pre.clear()

        wrong_record=[num2tagets[x] for x in wrong_record]
        false_classfy=[num2tagets[x]for x in false_classfy]
        wrong_pd=pd.DataFrame({'wrong label':wrong_record,'true label':false_classfy})
        print(wrong_pd[wrong_pd['wrong label']!='Other'])
        print(wrong_pd.value_counts())
        print("fasttext准确率:",model_correct[0]/(total_rows*0.2))

    def lstm_baseline(self,test_dataloader):
        lstm = torch.load('lstm.pth')

        model_correct = [0, 0, 0]
        vote_correct = 0
        wrong_record = []
        false_classfy = []

        for batch_idx, (inputs, label, binary_label) in enumerate(test_dataloader):
            for key in inputs.keys():
                inputs[key] = inputs[key].cuda()
            pre = []
            model = lstm
            model = model.cuda()
            model.eval()
            pred = model(inputs['input_ids']).squeeze()
            pred_num = pred.detach().cpu().numpy()
            correct = (pred_num == label.cpu().numpy()).sum()
            model_correct[0] += correct

            batch_wrong = []
            for i in range(len(pred_num)):
                if pred_num[i] != label.cpu().numpy()[i]:
                    wrong_record.append(label.cpu().numpy()[i])
                    false_classfy.append(pred_num[i])

            pre.append(pred_num)

            vote_pre = np.array(pre)
            result = [Counter(vote_pre[:, i]).most_common(1)[0][0] for i in range(len(label))]
            vote_correct += (result == label.detach().cpu().numpy()).sum()
            pre.clear()

        wrong_record = [num2tagets[x] for x in wrong_record]
        false_classfy = [num2tagets[x] for x in false_classfy]
        wrong_pd = pd.DataFrame({'wrong label': wrong_record, 'true label': false_classfy})
        print(wrong_pd[wrong_pd['wrong label'] != 'Other'])
        print(wrong_pd.value_counts())
        print("fasttext准确率:", model_correct[0] / (total_rows * 0.2))

    def train_fasttext(self):

        fasttext=fast_text()
        epoch = 20
        optimizer = torch.optim.Adam(fasttext.parameters(), lr=0.001)

        for e in range(epoch):
            for idx, batch in enumerate(dataloader):
                # 送入device
                inputs, label, binary_label = batch
                label = label.cuda()
                for key in inputs.keys():
                    inputs[key] = inputs[key].cuda()

                optimizer.zero_grad()
                fasttext = fasttext.cuda()

                fasttext.train()

                input = inputs['input_ids']
                loss = fasttext(input, label)
                print('epoch:', e, 'batch:', idx, 'loss:', loss.item())
                loss.backward()
                optimizer.step()
            torch.save(fasttext,'fasttext.pth')

    def train_lstm(self):

        lstm = LSTM_model()
        epoch = 20
        batch_size = 64
        optimizer = torch.optim.Adam(lstm.parameters(), lr=0.001)

        for e in range(epoch):
            for idx, batch in enumerate(dataloader):
                # 送入device
                inputs, label, binary_label = batch
                label = label.cuda()
                for key in inputs.keys():
                    inputs[key] = inputs[key].cuda()

                optimizer.zero_grad()
                lstm = lstm.cuda()

                lstm.train()

                input = inputs['input_ids']
                loss = lstm(input, label)
                print('epoch:', e, 'batch:', idx, 'loss:', loss.item())
                loss.backward()
                optimizer.step()
            torch.save(lstm, 'lstm.pth')

    def train_bert(self):
        epoch = 3
        model=Bert_model()

       # model=BertForSequenceClassification.from_pretrained(r"E:\python\pre_train_model\bert_file",num_labels=len(targets))


        peft_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            inference_mode=False,
            r=8,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["query", "value"]
        )

        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
        focal_loss=FocalLoss()

        for e in range(epoch):
            for idx, batch in enumerate(dataloader):
                # 送入device
                inputs, label, binary_label = batch
                label = label.cuda()
                for key in inputs.keys():
                    inputs[key] = inputs[key].cuda()
                model = model.cuda()
                optimizer.zero_grad()
                model.train()
                # logits=model(**inputs).logits
                # loss = focal_loss(logits,label)

                loss=model(inputs,label)

                print('epoch:', e, 'batch:', idx, 'loss:', loss.item())
                loss.backward()
                optimizer.step()
            #只保存新添加的参数
            save_params={
                k:v.to('cpu')
                for k,v in model.named_parameters() if v.requires_grad
            }
            torch.save(save_params, 'peft.pth')

    def vote(self,models):

        model_correct = [0, 0, 0]
        vote_correct = 0
        wrong_record = []
        false_classfy = []

        y_true=[]

        y_pred=[]

        for batch_idx, (inputs, label, binary_label) in enumerate(test_dataloader):
            for key in inputs.keys():
                inputs[key] = inputs[key].cuda()
            pre = []
            for i in range(3):
                model =models[i]
                model = model.cuda()
                model.eval()
                if i<2:
                    pred = model(inputs['input_ids']).squeeze()
                    pred_num = pred.detach().cpu().numpy()
                    correct = (pred_num == label.cpu().numpy()).sum()
                    model_correct[i] += correct

                else:
                    pred=torch.argmax(torch.softmax(model(**inputs).logits,dim=-1),dim=-1)
                    pred_num=pred.detach().cpu().numpy()
                    correct = (pred_num == label.cpu().numpy()).sum()
                    model_correct[i] += correct
                pre.append(pred_num)

            vote_pre = np.array(pre)
            result = [Counter(vote_pre[:, i]).most_common(1)[0][0] for i in range(len(label))]

            y_pred.extend(result)
            y_true.extend(label.cpu().numpy().tolist())

            vote_correct += (result == label.detach().cpu().numpy()).sum()
            pre.clear()

        wrong_pd = pd.DataFrame({'wrong label': wrong_record, 'true label': false_classfy})
        print(wrong_pd[wrong_pd['wrong label'] != 'Other'])
        print(wrong_pd.value_counts())
        print("fasttext准确率:", model_correct[0] / (total_rows * 0.2))
        print("lstm准确率:", model_correct[1] / (total_rows * 0.2))
        print("bert准确率:", model_correct[2] / (total_rows * 0.2))
        print("vote准确率:", vote_correct / (total_rows * 0.2))


        y_true=np.array(y_true)
        y_pred=np.array(y_pred)

        cm=cmx(y_true,y_pred)

        confusion_matrix=pd.DataFrame(cm,index=[i for i in num2tagets.values()],columns=[i for i in num2tagets.values()])

        print('Weighted precision', precision_score(y_true, y_pred, average='weighted'))
        print('Weighted recall', recall_score(y_true, y_pred, average='weighted'))
        print('Weighted f1-score', f1_score(y_true, y_pred, average='weighted'))

if __name__ == '__main__':
    #为数据添加数值型label
    train_data,num2tagets=apply_targets(train_data,targets)

    total_rows = train_data.shape[0]

    # 计算前4/5行的行数和后1/4行的行数
    first_part_rows = int(total_rows * 0.8)
    last_part_rows = total_rows - first_part_rows

    # 切分DataFrame
    first_part = train_data.iloc[:first_part_rows]
    last_part = train_data.iloc[-last_part_rows:]


    tokenizer=BertTokenizer.from_pretrained(r"E:\python\pre_train_model\bert_file\vocab.txt")


    Ds=DataStore(tokenizer,first_part,mode=True)
    Ds_test=DataStore(tokenizer,last_part,mode=False)


    dataloader=DataLoader(Ds,batch_size=64,shuffle=True)
    test_dataloader=DataLoader(Ds_test,batch_size=38,shuffle=False)

    model_chain=model_use()
  #  model_chain.train_bert()
    lstm=torch.load('lstm.pth')
    fasttext=torch.load('fasttext.pth')
    peft_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            inference_mode=False,
            r=8,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["query", "key", "value"]
        )


    bert=BertForSequenceClassification.from_pretrained(r"E:\python\pre_train_model\bert_file",num_labels=len(targets))
    bert_model=get_peft_model(bert,peft_config)

    state_dict=bert_model.state_dict()
    state_dict.update(torch.load('peft.pth'))

    bert_model.load_state_dict(state_dict)
    model_chain.vote([lstm,fasttext,bert_model])

