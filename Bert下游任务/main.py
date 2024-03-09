import torch.optim
import matplotlib.pyplot as plt
from Config import  Config
from loader import Make_data
from torch.utils.data import DataLoader
import torch.nn as nn
from model import BertClassifier
from transformers import BertTokenizer

torch.cuda.manual_seed_all(999)

def evaluate(dataloader,model):

    dataloader.mode='test'
    model.eval()
    with torch.no_grad():
        right = 0
        wrong = 0

        for idx, batch in enumerate(dataloader):
            intput_ids, mask, type_id, label = batch
            intput_ids, mask, type_id, label = intput_ids.cuda(), mask.cuda(), type_id.cuda(), label.cuda()
            pool_output = model(intput_ids, mask, type_id)
            pred = torch.argmax(pool_output, dim=1)

            for i in range(len(pred)):
                if pred[i] == label[i]:
                    right += 1
                else:
                    wrong += 1
    print('正确率:', right / (right + wrong))

def train():
    bert_model = BertClassifier(Config)
    bert_model = bert_model.cuda()
    optimizer = torch.optim.Adam(bert_model.parameters(), lr=Config['lr'])
    dataset = Make_data(Config)
    dataloader = DataLoader(dataset=dataset, batch_size=Config['batch_size'], shuffle=True, drop_last=True)
    train_loss = []
    for E in range(Config['epoch']):
        for idx, bat in enumerate(dataloader):
            optimizer.zero_grad()
            intput_ids, mask, type_id, label = bat
            intput_ids, mask, type_id, label = intput_ids.cuda(), mask.cuda(), type_id.cuda(), label.cuda()
            pool_output = bert_model(intput_ids, mask, type_id)
            loss = bert_model.compute_loss(pool_output, label)
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
            print('Epoch:{},iter：{},loss:{}'.format(E, idx, loss.item()))
        evaluate(dataloader, bert_model)

    torch.save(bert_model, 'bert_classifier.pth')

    plt.plot(train_loss)
    plt.show()


def query():
    q=input('请输入评论:')

    model=torch.load('bert_classifier.pth')
    tokenizer=BertTokenizer(Config['vocab_path'])

    output = tokenizer.encode_plus(q, max_length=Config['max_len'],truncation=True,padding='max_length')

    input_ids=output['input_ids']
    mask=output['attention_mask']
    type_ids=output['token_type_ids']
    input_ids=torch.LongTensor([input_ids])
    mask=torch.LongTensor([mask])
    type_ids=torch.LongTensor([type_ids])

    input_ids=input_ids.cuda()
    mask=mask.cuda()
    type_ids=type_ids.cuda()

    output=model(input_ids,mask,type_ids)
    res=torch.argmax(output,dim=1)

    print('好评' if res.item()==1 else '差评')



if __name__ == '__main__':
    query()


