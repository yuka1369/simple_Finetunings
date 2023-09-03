import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from torch.utils.data import DataLoader, Dataset
import sys
import os
import json
import pandas as pd
import torch.nn as nn
import matplotlib.pyplot as plt
import plotly.express as px
from evaluation import evaluate_gen_redial

#cuda指定
#device_str = 'cuda:0'
#device = torch.device(device_str)

# 事前学習済みのGPT-2モデルとトークナイザを読み込む
model_name = "gpt2-medium"  # モデルの名前に応じて適宜変更
model = GPT2LMHeadModel.from_pretrained(model_name)

tokenizer = GPT2Tokenizer.from_pretrained(model_name)
#tokenizer.add_special_tokens({'pad_token': '[PAD]'})

#print(f"tokenizer.get_vocab(){tokenizer.get_vocab()}")
"""
print(f"tokenizer.all_special_tokens:{tokenizer.all_special_tokens}")
tokenizer.all_special_tokens:['<|endoftext|>']
"""

# ファインチューニング用のテキストデータを読み込む
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

# 読み込みたいファイルのパスを組み立てる
train_data_path = os.path.join(parent_dir, "data", "train_org.json")  # 親ディレクトリ内のdataディレクトリ内のfile.txtを指定

# ファイルを読み込む
f = open(train_data_path)
data = f.read()
data = json.loads(data)

# データを分割する割合を設定
train_ratio = 0.7  # トレーニングデータの割合
val_ratio = 0.15  # 検証データの割合
test_ratio = 0.15  # テストデータの割合

# データを分割
total_samples = len(data)
train_split = int(train_ratio * total_samples)
val_split = int((train_ratio + val_ratio) * total_samples)

# train_data = data[:train_split]
# val_data = data[train_split:val_split]
# test_data = data[val_split:]

#お試し用
train_data = data[:3]
val_data = data[11:14]
test_data = data[15:18]



for dt in data:
    """
    dt["context"]
    --
    ["Hi there, how are you? I'm looking for movie recommendations"]
    """

    contexts = ' '.join(dt["context"])
    
    ut = dt["utterance"]

    # [SEP] のトークンid 685, 5188, 47, 60,　だけど，登録した方がいいか？まぁいいかとりあえずしなくて．
    contexts_ut = contexts +" [SEP] " + ut + '<|endoftext|>'
    dt["context"] = contexts_ut
    #dt["context"] = tokenizer(dt["context"])
    #print(dt["context"])



#print(f"data[0]:{data[0]}")
#{'context': "Hi there, how are you? I'm looking for movie recommendations [SEP] I am doing okay. What kind of movies do you like?<|endoftext|>", 'utterance': 'I am doing okay. What kind of movies do you like?', 'mentioned': [], 'node_candidate1': [30452, 30453, 30454, 30455, 30456, 30457], 'label_1': [3], 'node_candidate2': [[30434, 30435, 30436, 30437, 30438, 30439, 30440, 30441, 30442, 30443, 30444, 30445, 30446, 30447, 30448, 30449, 30450, 30451, 30458]], 'label_2': [[]], 'intent': 'question', 'new_mentioned': [30455], 'dialog_num': 0, 'system_turn': 0, 'label_rec': [3]}
#{'context': "Hi there, how are you? I'm looking for movie recommendations [SEP] I am doing okay. What kind of movies do you like?<|endoftext|>", 'utterance': 'I am doing okay. What kind of movies do you like?', 'mentioned': [], 'node_candidate1': [30452, 30453, 30454, 30455, 30456, 30457], 'label_1': [3], 'node_candidate2': [[30434, 30435, 30436, 30437, 30438, 30439, 30440, 30441, 30442, 30443, 30444, 30445, 30446, 30447, 30448, 30449, 30450, 30451, 30458]], 'label_2': [[]], 'intent': 'question', 'new_mentioned': [30455], 'dialog_num': 0, 'system_turn': 0, 'label_rec': [3]}



class MyDataset(torch.utils.data.Dataset):
    def __init__(self, data, tokenizer, max_length):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    # len()を使用すると呼ばれる
    def __len__(self):
        return len(self.data)

    # 要素を参照すると呼ばれる関数
    def __getitem__(self, idx):
        text = self.data[idx]['context']
        #print(f"text:{text}")
        input_ids = self.tokenizer.encode(text, max_length=self.max_length, truncation=True)
        #input_ids = self.tokenizer.encode(text, add_special_tokens=True, max_length=self.max_length, truncation=True, padding='max_length')
        #input_ids = self.features_values = data['context']['input_ids']
        #print(f"input_ids:{input_ids}")
        """
        input_ids:[31, 1433, 15277, 21, 632, 318, 257, 3621, 3807, 13, 1867, 910, 30, 2488, 44675, 1120, 8920, 257, 43937, 513, 35, 1998, 7620, 329, 502, 314, 1107, 8288, 326, 3807, 475, 4398, 470, 1775, 340, 287, 257, 981, 13, 4231, 345, 257, 16738, 6882, 380, 22281, 4336, 30, 1400, 13, 314, 716, 407, 13, 314, 588, 20681, 12, 12463, 6918, 13, 8192, 345, 7342, 2488, 1314, 3270, 3388, 13, 2011, 1194, 4004, 20681, 12, 12463, 6918, 389, 2488, 3720, 19504, 290, 2488, 5705, 8298, 1400, 314, 423, 407, 1775, 326, 530, 475, 423, 1775, 477, 262, 2488, 1238, 11785, 17, 6918, 13, 2141, 345, 588, 883, 3363, 11, 314, 1842, 2488, 1238, 11785, 17, 2168, 13, 554, 1109, 314, 716, 6568, 546, 262, 7865, 2488, 1238, 11785, 17]
        """
        return torch.tensor(input_ids)

#テスト用20230930には削除
evaluate_gen_redial(val_data)


#####Train####
# データローダーの設定
print(f"Train----------------------")

batch_size = 1
max_length = 512

dataset = MyDataset(train_data, tokenizer, max_length)
#print(f"dataset[0]:{dataset[0]}")
#data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
#criterion = nn.CrossEntropyLoss()
criterion = nn.BCEWithLogitsLoss(reduction="sum")

num_epochs = 2  # 適宜調整
loss_list = []
for epoch in range(num_epochs):
    model.train()
    for batch in data_loader:
        print(f"batch:{batch}")
        batch = batch.to(device)
        #print(f"batch.device:{batch.device}")

        outputs = model(batch,labels=batch)  # 入力とターゲットが同じと仮定
        #loss = criterion(batch, labels=batch)
        loss = outputs.loss
        print(f"grobal_loss:{loss}")
        loss_list.append(float(loss))

        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        if num_epochs == 100:
            df_train_loss = pd.DataFrame(loss_list)
            fig = px.line(df_train_loss)
            #fig = px.line(df_train_loss, x='epoch', y='loss')
            fig.show()
            fig.write_html("train_loss_log/train_loss_tmp.html")
            fig.write_image("train_loss_log/train_loss_tmp.png") 

            save_data_path = os.path.join(parent_dir, "saved", str(num_epochs)+"savedRedialGPT"+".pt")
            torch.save(model.state_dict(),save_data_path )

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")


#自前のトークナイザーを保存用.コード途中．現時点では元のgpt-2のトークナイザーから変更してないため保留
# #save_data_path = os.path.join(parent_dir, "saved", str(num_epochs)+"savedRedialGPT_tork.json")
# torch.save(model.state_dict(),save_data_path )
# tokenizer.save_pretrained(save_data_path)




#####Valid####
print(f"Valid---------------------")
dataset = MyDataset(val_data, tokenizer, max_length)
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
val_loss_list = []
#bleu_array = []
for epoch in range(num_epochs):
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in data_loader:
            batch = batch.to(device)
            print(f"batch:{batch}")
            outputs = model(batch,labels=batch)
            #loss = criterion(batch,labels=batch)
            loss = outputs.loss
            val_loss_list.append(loss)
            val_loss += loss.item()
            #_, predicted = torch.max(outputs, 1)
            predicted = outputs.logits.argmax(dim=-1)
            print(f"predicted:{predicted}")
            total += batch.size(0)
            correct += (predicted == batch).sum().item()

            if num_epochs == 2:
                df_val_loss = pd.DataFrame(val_loss_list)
                fig = px.line(df_val_loss)
                #fig = px.line(df_train_loss, x='epoch', y='loss')
                fig.show()
                fig.write_html("val_loss_log/train_loss_tmp.html")
                fig.write_image("val_loss_log/train_loss_tmp.png") 

    val_accuracy = correct / total
    print(f'Epoch [{epoch+1}/{num_epochs}] - Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}')

evaluation(val_data)



#####Test####
print(f"Test---------------------")
dataset = MyDataset(test_data, tokenizer, max_length)
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
for epoch in range(num_epochs):
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in data_loader:
            batch = batch.to(device)
            outputs = model(batch,labels=batch)
            #loss = criterion(batch,labels=batch)
            loss = outputs.loss
            test_loss += loss.item()
            #_, predicted = torch.max(outputs, 1)
            predicted = outputs.logits.argmax(dim=-1)
            total += batch.size(0)
            correct += (predicted == batch).sum().item()

    test_accuracy = correct / total
    print(f'Epoch [{epoch+1}/{num_epochs}] - Validation Loss: {test_loss:.4f}, Validation Accuracy: {test_accuracy:.4f}')



