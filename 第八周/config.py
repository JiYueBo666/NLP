config={
    'model':"edit_match",
    'lr':0.001,
    "optimizer":"Adam",
    "epoch":100,
    "batch_size":32,
    "max_len":30,
    "embedding_dim":128,
    'hidden_dim':128,
    "train_path":r"E:data\train.json",
    "valid_path":r"E:data\valid.json",
    "schema_path":r"E:data\schema.json",
    "positive_rate":0.5,#正负样本比例
    "epoch_data_size": 1000,  # 每轮训练中采样数量
    "out_dim":128,
    "vocab_path":r"E:\python\bert_file\vocab.txt"
}
