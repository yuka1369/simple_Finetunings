# -*- coding: utf-8 -*-
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, pipeline
import os
import sys

class Generator():
    def __init__(self):
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #トークナイザ．自前の変更を加えておらず"gpt2-medium"のままなのでこのまま
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2-medium")
        #configで読み込み　"gpt2-medium"と変えてないためこのまま
        self.model= GPT2LMHeadModel.from_pretrained("gpt2-medium")
        self.model.to(device)
        self.model.eval()
        # 学習した重みの読み込み
        parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        sys.path.append(parent_dir)
        saved_data_path = os.path.join(parent_dir, "saved", "2savedRedialGPT.pt")  # 親ディレクトリ内のdataディレクトリ内のfile.txtを指定

        #これでいいのか不安
        self.model.load_state_dict(torch.load(saved_data_path))

        

    def generate(self,raw_text):
        # テキスト生成のためのパイプラインを設定
        generator = pipeline("text-generation", model=self.model, tokenizer=self.tokenizer)
        # # テキスト生成のデモ
        prompt = raw_text
        # output_text = generator(prompt, max_length=100, num_return_sequences=1, temperature=0.7, return_special_tokens=True)[0]['generated_text']

        # print(output_text)

        output = generator(prompt, max_length=100, num_return_sequences=1, temperature=0.7, return_special_tokens=True)
        #print(f"output:{output}")

        generated_text = output[0]['generated_text']
        token_ids = self.tokenizer.encode(generated_text, add_special_tokens=False)
        #ぜんぜんスペシャルトークンの<|endoftext|>:50256も[SEP]も生成されない．ファインチューニングできてないのか

        #print("Generated Text:")
        #print(generated_text)

        #print("\nToken IDs:")
        #print(token_ids)

        return generated_text

# if __name__ == '__main__':
#     gen = Generator()
#     gen_text = gen.generate("What your favorite movie?")
#     print(f"gen_text:{gen_text}")