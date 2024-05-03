import torch

config={
    'train_path':r"E:\算法实验\Bert下游任务-NER\china-people-daily-ner-corpus\example.train.txt",
    'test_path':r"E:\算法实验\Bert下游任务-NER\china-people-daily-ner-corpus\example.test.txt",
    'valid_path':r"E:\算法实验\Bert下游任务-NER\china-people-daily-ner-corpus\example.dev.txt",
    'vocab_path':r"E:\python\pre_train_model\bert_file\vocab.txt",
    'bert_path':r"E:\python\pre_train_model\bert_file",
    'save_path':"bert_classfy_path",
    'max_len':128,#最大句子长度
    'hidden_size':512,
    'batch_size':128,
    'epoch':50,
    'lr':0.0001,
    'num_labels':7,
    'cuda':True,
    'table_size':4377
}

