import torch
from transformers import BertTokenizer
from Config import config
from loader import Dataset
from model import BertNER
from torch.utils.data import DataLoader
import torch.optim as optim

def get_key(query_list,my_dict):
    '''
    :param query_list: 预测类别列表
    :param my_dict:
    :return:
    '''
    res=[]
    for val in query_list:
        for key, value in my_dict.items():
             if val == value:
                res.append(key)
                break
    return res

def get_idx(target_list):
    idx_list=[]
    for i in range(len(target_list)):
        if target_list[i]!='O':
            idx_list.append(i-1)
    return idx_list

def get_single_key(idx,my_dict):
    for key,value in my_dict.items():
        if idx==value:
            return key

def train(dataloader):

    model = BertNER(config)
    model=model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=config['lr'])
    optimizer = optimizer

    for E in range(config['epoch']):
        for idx, batch in enumerate(dataloader):
            optimizer.zero_grad()

            input_ids, mask, type_ids, label = batch
            input_ids, mask, type_ids, label = input_ids.cuda(), mask.cuda(), type_ids.cuda(), label.cuda()
            loss, logits = model(input_ids, mask, type_ids, label)

            loss.backward()
            optimizer.step()

            print("Epoch: {}, Batch: {}, Loss: {}".format(E, idx, loss.item()))
    torch.save(model, 'NERbert.pth')

def compute_correct(logits,labels):
    ACC=0
    for pred,label in zip(logits, labels):
        right = 0
        wrong = 0
        for i in range(len(pred)):
            if pred[i] == label[i]:
                right+= 1
        ACC+=right/len(pred)
    ACC=ACC/len(labels)
    return ACC

def evaluate(dataloader,model):

    for idx,batch in enumerate(dataloader):
        input_ids, mask, type_ids, label = batch
        input_ids, mask, type_ids, label = input_ids.cuda(), mask.cuda(), type_ids.cuda(), label.cuda()

        loss, logits = model(input_ids, mask, type_ids, label)

        #logits:[batch_size,sent_len,label_len]

        logits=torch.argmax(logits,dim=-1)

        logits=logits.detach().cpu().tolist()
        label=label.detach().cpu().tolist()

        ACC=compute_correct(logits,label)
        print("Batch平均准确率:",ACC)

def query(model,query_dict):

    sentence=input("输入文本:")

    tokenizer=BertTokenizer(config['vocab_path'])

    output=tokenizer.encode_plus(sentence,max_length=config['max_len'],padding='max_length',truncation=True)

    input_ids, mask, type_ids=output['input_ids'],output['attention_mask'],output['token_type_ids']

    input_ids=torch.LongTensor([input_ids])
    mask=torch.LongTensor([mask])
    type_ids=torch.LongTensor([type_ids])
    input_ids=input_ids.cuda()
    mask=mask.cuda()
    type_ids=type_ids.cuda()
    loss,logits=model(input_ids,mask,type_ids)
    logits=torch.argmax(logits,dim=-1)

    print(logits)

    #获取分类类别
    logits=logits.detach().cpu().tolist()
    name=""
    org=""
    gpe=""
    for idx in range(len(logits[0])):
        if "PER" in get_single_key(logits[0][idx],query_dict):
            #sentence[0]是补充的cls
            name+=sentence[idx-1]
        elif "ORG" in get_single_key(logits[0][idx],query_dict):
            org+=sentence[idx-1]
        elif "GPE" in get_single_key(logits[0][idx],query_dict):
            gpe+=sentence[idx-1]
    print("识别到的人名:",name)
    print("识别到的机构:",org)
    print("识别到的地名:",gpe)


if __name__ == '__main__':

    dataset=Dataset(config)
    dataloader=DataLoader(dataset,batch_size=config['batch_size'],shuffle=True)
    dataset.load_test()
    test_loader=DataLoader(dataset,batch_size=config['batch_size'],shuffle=True)

 #   train(dataloader)
    model=torch.load('NERbert.pth')
    model=model.cuda()
 #   evaluate(test_loader,model)

    query(model,dataset.class_type)