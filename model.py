import math
import torch
import numpy as np
import torch.nn as nn
from Config import config
from transformers import BertModel,BertTokenizer
from loader import Dataset
from torch.utils.data import DataLoader
from torchcrf import CRF
import random
import numpy as np
from torch.nn import LSTM
def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

seed_everything(100)


class Flat(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.config=config

        self.embedding=nn.Embedding(config['table_size'],config['hidden_size'])
        self.d_model=config['hidden_size']

        self.lstm = LSTM(input_size=self.d_model,hidden_size=self.d_model, num_layers=1, batch_first=True, bidirectional=True)
        self.dense=nn.Linear(self.d_model*2,self.d_model)



        self.position_linear=nn.Linear(self.config['max_len'],self.d_model)
        self.relu=nn.ReLU()
        self.num_labels=self.config['num_labels']
        self.crf = CRF(num_tags=self.num_labels, batch_first=True)

        self.q_linear=nn.Linear(self.d_model,self.d_model)
        self.k_linear=nn.Linear(self.d_model,self.d_model)
        self.v_linear=nn.Linear(self.d_model,self.d_model)


        self.ffn1=nn.Linear(self.d_model,self.d_model)
        self.ffn2=nn.Linear(self.d_model,self.d_model)


        self.LN=nn.LayerNorm(self.d_model)

        self.softmax=nn.Softmax(dim=-1)

        self.classify_layer=nn.Linear(self.d_model,self.config['num_labels'])
        self.classify_loss=nn.CrossEntropyLoss(ignore_index=-1)
        if self.config['cuda']:
            self.classify_loss=self.classify_loss.cuda()

    def P_span(self,d):
        batch,k,_=d.shape

        pos_enc = torch.zeros_like(d)
        if self.config['cuda']:
            pos_enc=pos_enc.cuda()


        # 计算奇数位置的编码
        div_term = torch.exp(torch.arange(0, k, 2, dtype=torch.float32) * (-torch.log(torch.tensor(10000.0)) /self.d_model))
        if self.config['cuda']:
            div_term=div_term.cuda()
        pos_enc[:, :, 0::2] = torch.sin(d[:,:,0::2] * div_term)

        # 计算偶数位置的编码
        div_term = torch.exp(torch.arange(1, k, 2, dtype=torch.float32) * (-torch.log(torch.tensor(10000.0)) /self.d_model))
        if self.config['cuda']:
            div_term = div_term.cuda()
        pos_enc[:, :, 1::2] = torch.cos(d[:,:,1::2] * div_term)

        return pos_enc

    def relative_position(self,ps,pe):

        d_hh= (ps.unsqueeze(1) - ps.unsqueeze(2)).transpose(-1,-2)
        d_tt=(pe.unsqueeze(1) - pe.unsqueeze(2)).transpose(-1,-2)
        d_ht=ps.unsqueeze(2) - pe.unsqueeze(1)
        d_th=pe.unsqueeze(2) - ps.unsqueeze(1)

        P_hh=self.P_span(d_hh).unsqueeze(1)
        P_tt=self.P_span(d_tt).unsqueeze(1)
        P_ht=self.P_span(d_ht).unsqueeze(1)
        P_th=self.P_span(d_th).unsqueeze(1)

        P=torch.cat([P_hh,P_th,P_ht,P_tt],dim=1)#[batch,4,sentence_len,sentence_len]
        P=torch.sum(P,dim=1).squeeze()
        P=P.to(torch.float32)

        #[batch,sentence_len,sentence_len]
        R=self.relu(self.position_linear(P))

        return R

    def self_attention(self,word_embeddings,R_scores):
        '''
        :param word_embeddings:[batch_size,sentence_len,d_model]
        :param R_scores: [batch_size,sentence_len,d_model]
        :return:
        '''


        Q=self.q_linear(word_embeddings+R_scores)
        K_T=self.k_linear(word_embeddings+R_scores).transpose(-1,-2)

        scores=self.softmax(torch.matmul(Q,K_T)/(math.sqrt(self.d_model)))

        output=torch.matmul(scores,word_embeddings)

        return output


    def residual_block(self,att_output,word_embeddings):
        Add=att_output+word_embeddings
        ln_res=self.LN(Add)
        return ln_res


    def FFN(self,inputs):
        return self.ffn2(self.relu(self.ffn1(inputs)))


    def forward(self,inputs,ps,pe,label=None):

        position_embedding=self.relative_position(ps,pe)


        #[batch,sentence_len,sentence_len,d_model]

        look_tabble=self.embedding(inputs)#[b,man_len,hidden]

        output,(h,_)=self.lstm(look_tabble)
        output=self.dense(output)

        atten_layer=self.self_attention(output,position_embedding)
        residual_layer1=self.residual_block(atten_layer,output)
        ffn_layer=self.FFN(residual_layer1)
        residual_layer2=self.residual_block(ffn_layer,residual_layer1)
        pred=self.classify_layer(residual_layer2)#[batch_size,sen,num_class]

        if label is not None:
            #loss1=self.classify_loss(pred.view(-1,self.num_labels),label.view(-1))

            crf_input=pred
            crf_label=label
            mask=crf_label.gt(-0.5)

            try:
                loss=-self.crf(crf_input,crf_label,mask=mask)

                return loss
            except Exception as e:
                print(crf_label)

        else:
            return self.crf.decode(pred)


