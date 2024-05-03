import torch
from transformers import BertTokenizer
from Config import config
from model import Flat
from loader import Dataset
from torch.utils.data import DataLoader
import torch.optim as optim
import matplotlib.pyplot as plt
import logging
import numpy as np
from collections import defaultdict
import random
import re

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

seed_everything(100)


class Evaluater:
    def __init__(self, config, model, logger,dataloader):
        self.config = config
        self.model = model
        self.logger = logger

        self.dataloader=dataloader

    def eval(self,epoch):
        self.logger.info("开始测试第%d轮模型效果：" % epoch)
        self.stats_dict = {"LOCATION": defaultdict(int),
                           "TIME": defaultdict(int),
                           "PERSON": defaultdict(int),
                           "ORGANIZATION": defaultdict(int)}


        self.model.eval()
        for idx,batch in enumerate(self.dataloader):
            sentence,inputs,ps,pe,label=batch
            if self.config['cuda']:
                inputs=inputs.cuda()
                ps=ps.cuda()
                pe=pe.cuda()
                label=label.cuda()
                with torch.no_grad():
                    pred=self.model(inputs,ps,pe)
                    self.write_stats(label, pred, sentence)
        self.show_stats()
        return

    def length(self,label):
        for i in range(len(label)):
            if label[i]==-1:
                return i-1
        return len(label)-1

    def write_stats(self, labels, pred_results, sentences):
        assert len(labels) == len(pred_results) == len(sentences)
        for true_label, pred_label, sentence in zip(labels, pred_results, sentences):

            valid_sentence=self.length(true_label)

            true_label = true_label.cpu().detach().tolist()
            true_entities = self.decode(sentence, true_label,valid_sentence)
            pred_entities = self.decode(sentence, pred_label,valid_sentence)

            # 正确率 = 识别出的正确实体数 / 识别出的实体数
            # 召回率 = 识别出的正确实体数 / 样本的实体数
            for key in ["PERSON", "LOCATION","ORGANIZATION"]:
                self.stats_dict[key]["正确识别"] += len(
                    [ent for ent in pred_entities[key] if ent in true_entities[key]])
                self.stats_dict[key]["样本实体数"] += len(true_entities[key])
                self.stats_dict[key]["识别出实体数"] += len(pred_entities[key])
        return
    def show_stats(self):
        F1_scores = []
        for key in ["PERSON", "LOCATION", "ORGANIZATION"]:
            # 正确率 = 识别出的正确实体数 / 识别出的实体数
            # 召回率 = 识别出的正确实体数 / 样本的实体数
            precision = self.stats_dict[key]["正确识别"] / (1e-5 + self.stats_dict[key]["识别出实体数"])
            recall = self.stats_dict[key]["正确识别"] / (1e-5 + self.stats_dict[key]["样本实体数"])
            F1 = (2 * precision * recall) / (precision + recall + 1e-5)
            F1_scores.append(F1)
            self.logger.info("%s类实体，准确率：%f, 召回率: %f, F1: %f" % (key, precision, recall, F1))
        self.logger.info("Macro-F1: %f" % np.mean(F1_scores))
        correct_pred = sum([self.stats_dict[key]["正确识别"] for key in ["PERSON", "LOCATION", "ORGANIZATION"]])
        total_pred = sum(
            [self.stats_dict[key]["识别出实体数"] for key in ["PERSON", "LOCATION", "ORGANIZATION"]])
        true_enti = sum([self.stats_dict[key]["样本实体数"] for key in ["PERSON", "LOCATION", "ORGANIZATION"]])
        micro_precision = correct_pred / (total_pred + 1e-5)
        micro_recall = correct_pred / (true_enti + 1e-5)
        micro_f1 = (2 * micro_precision * micro_recall) / (micro_precision + micro_recall + 1e-5)
        self.logger.info("Micro-F1 %f" % micro_f1)
        self.logger.info("--------------------")
        return
    def decode(self,sentence,labels,valid_sentence):

        sentence=sentence[:valid_sentence+1]
        labels = labels[:valid_sentence+1]

        labels = "".join([str(x) for x in labels[:len(sentence)]])
        results = defaultdict(list)
        for location in re.finditer("(12+)", labels):
            s, e = location.span()
            results["LOCATION"].append(sentence[s:e])
        for location in re.finditer("(56+)", labels):
            s, e = location.span()
            results["ORGANIZATION"].append(sentence[s:e])
        for location in re.finditer("(34+)", labels):
            s, e = location.span()
            results["PERSON"].append(sentence[s:e])
        return results
    def recognize(self,dataset):

        results = defaultdict(list)
        sentence=input("输入句子:")

        text_encode,ps,pe=dataset.encode(sentence)
        text_encode=torch.tensor([text_encode]).cuda()
        ps=torch.tensor([ps]).cuda()
        pe=torch.tensor([pe]).cuda()

        pred=self.model(text_encode,ps,pe)[0]
        pred = "".join([str(x) for x in pred])

        for location in re.finditer("(12+)",pred):
            s, e = location.span()
            results["LOCATION"].append(sentence[s:e])
        for location in re.finditer("(56+)", pred):
            s, e = location.span()
            results["ORGANIZATION"].append(sentence[s:e])
        for location in re.finditer("(34+)", pred):
            s, e = location.span()
            results["PERSON"].append(sentence[s:e])


        print("识别的机构:",results["ORGANIZATION"])
        print("识别的地点:",results["LOCATION"])
        print("识别的人名:",results["PERSON"])

def train():
    Epoch = config['epoch']
    model=Flat(config)
    model=torch.load('model.pth')
    if config['cuda']:
        model=model.cuda()

    TrainDataset = Dataset(config)
    TrainDataloader =DataLoader(dataset=TrainDataset,batch_size=config['batch_size'],shuffle=True)

    evaluater = Evaluater(config, model, logger,TrainDataloader)


    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])

    for e in range(Epoch):
        model.train()
        for idx, batch in enumerate(TrainDataloader):
            optimizer.zero_grad()
            _,input,ps,pe,label = batch

            if config['cuda']:
                input=input.cuda()
                ps=ps.cuda()
                pe=pe.cuda()
                label=label.cuda()
            loss = model(input, ps, pe, label)
            loss.backward()
            optimizer.step()

            print('Epoch:{},batch:{},  loss: {},'.format(e + 1, idx,round(loss.item(), 2)))
        evaluater.eval(e)
        torch.save(model, 'model.pth')



if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
   # train()
    model = Flat(config)
    model=torch.load('model.pth')
    ValidDataset = Dataset(config)
    ValidDataset.do_train=False
    ValidDataloader =DataLoader(dataset=ValidDataset,batch_size=config['batch_size'],shuffle=True)
    evaluater = Evaluater(config, model, logger, ValidDataloader)
    evaluater.recognize(ValidDataset)
    #evaluater.eval(1)