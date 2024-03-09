import torch.nn as nn
import torch
from transformers import BertModel

class BertClassifier(nn.Module):
    def __init__(self, config,num_classes=2):
        super(BertClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(config['bert_path'])
        #self.tokenizer=self.bert.
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(768, num_classes)
        self.loss=nn.CrossEntropyLoss()
        self.softmax=nn.Softmax(dim=1)

    def compute_loss(self,pool_output,labels):
        self.loss=self.loss.cuda()
        loss_value=self.loss(pool_output,labels.squeeze(1))
        return loss_value




    def forward(self, input_ids, attention_mask,input_type):
        output=self.bert(input_ids=input_ids,attention_mask=attention_mask,token_type_ids=input_type)
        cate_=output.pooler_output
        classfy=self.fc(self.dropout(cate_))
        # if input_ids.shape[0]==1:
        #     classfy=self.softmax(classfy)
        return classfy
