# 导入必要的库
import math
from typing import List, Optional, Tuple, Union
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
import os
import pandas as pd

from torch.utils.data import IterableDataset, Dataset
import json
import numpy as np
from transformers import  PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers import PretrainedConfig
from transformers import Trainer, TrainingArguments, AutoModelForCausalLM, AutoTokenizer, DefaultDataCollator, DataCollatorForTokenClassification, AutoConfig


# 预训练数据集
class LLMDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_seq_len):
        """
        预训练数据集初始化.
        :param data_path: 数据路径.
        :param tokenizer: 分词器.
        :param max_seq_len: 最大序列长度.
        """
        super().__init__()
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        with open(self.data_path, 'r', encoding='utf-8') as f:
            self.data = f.readlines()
    
    def __len__(self):
        """
        返回数据集长度.
        """
        return len(self.data)
    
    def __getitem__(self, index: int):
        """
        获取单个样本.
        :param index: 样本索引.
        :return: 样本.
        """
        
        line = self.data[index]
        line = json.loads(line)
        text = '<s>' + line['text'] + '</s>'
        input_ids = self.tokenizer.encode(text)
        text_len = len(input_ids)
        if text_len > self.max_seq_len:
            input_ids = input_ids[:self.max_seq_len]
        else:
            input_ids = input_ids + [0] * (self.max_seq_len - text_len)
        input_ids = np.array(input_ids)
        X = np.array(input_ids[:-1]).astype(np.int64)
        Y = np.array(input_ids[1:]).astype(np.int64)
        
        # 将padding部分的标签设置为-100，以便在计算loss时忽略
        if text_len < self.max_seq_len:
            Y[text_len-1:] = -100
            
        return {
            'input_ids': torch.from_numpy(X),
            'labels': torch.from_numpy(Y),
        }
        
# # SFT数据集 # up用的
# class SFTDataset(Dataset):
#     def __init__(self, data_path, tokenizer, max_seq_len):
#         """
#         SFT数据集初始化.
#         :param data_path: 数据路径.
#         :param tokenizer: 分词器.
#         :param max_seq_len: 最大序列长度.
#         """
#         super().__init__()
#         self.data_path = data_path
#         self.tokenizer = tokenizer
#         self.max_seq_len = max_seq_len
        
#         with open(self.data_path, 'r', encoding='utf-8') as f:
#             self.data = f.readlines()
            
#     def __len__(self):
#         """
#         返回数据集长度.
#         """
#         return len(self.data)    
    
#     def __getitem__(self, index):
#         """
#         获取单个样本.
#         :param index: 样本索引.
#         :return: 样本.
#         """
#         line = self.data[index]
#         line = json.loads(line)
#         instruction_text = line['instruction']
#         input_text = line['input']
#         output_text = line['output']
#         history = line['history']
#         query = instruction_text + input_text
#         answer = output_text + self.tokenizer.eos_token
#         messages = []
#         if history:
#             for i in history:
#                 messages.append({'role': 'user', 'content': i[0]})
#                 messages.append({'role': 'assistant', 'content': i[1]})
        
#         messages.append({'role': 'user', 'content': query})   
#         prompt = self.tokenizer.apply_chat_template(messages, tokenize=False) 
#         prompt_input_ids = self.tokenizer.encode(prompt)
#         answer_input_ids = self.tokenizer.encode(answer)
#         input_ids = prompt_input_ids + answer_input_ids
#         labels = [0] * len(prompt_input_ids) + answer_input_ids
#         text_len = len(input_ids)
#         if text_len > self.max_seq_len:
#             input_ids = input_ids[:self.max_seq_len]
#             labels = labels[:self.max_seq_len]
#         else:
#             input_ids = input_ids + [0] * (self.max_seq_len - text_len)
#             labels = labels + [0] * (self.max_seq_len - text_len)
        
#         input_ids = input_ids[:-1]
#         labels = labels[1:]
#         return {'input_ids': torch.tensor(input_ids), 'labels': torch.tensor(labels)}

import json
import torch
from torch.utils.data import Dataset

class SFTDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_seq_len):
        """
        SFT数据集初始化 (支持 ChatML 格式)
        :param data_path: 数据路径 (.jsonl 文件).
        :param tokenizer: 分词器.
        :param max_seq_len: 最大序列长度.
        """
        super().__init__()
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.ignore_index = -100  # PyTorch Loss计算时忽略的标签ID
        
        # 加载数据
        with open(self.data_path, 'r', encoding='utf-8') as f:
            self.data = [json.loads(line) for line in f]

    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, index):
        sample = self.data[index]
        
        # 1. 获取对话列表
        # 兼容提供的两种格式：如果是text格式直接用，如果是conversations格式则解析
        if 'conversations' in sample:
            messages = sample['conversations']
        elif 'text' in sample:
            # 如果已经是处理好的文本，这里可能需要根据你的逻辑调整
            # 这里假设text已经是格式化好的，但SFT通常需要区分角色做mask
            # 为了演示SFT的核心逻辑，这里主要处理 conversations 格式
            return self._process_text_only(sample['text'])
        else:
            raise ValueError("数据格式错误，未找到 'conversations' 或 'text' 字段")

        # 2. 构建 Input IDs 和 Labels
        # 采用 ChatML 格式: <|im_start|>role\ncontent<|im_end|>\n
        input_ids = []
        labels = []
        
        # 这里的特殊token需要根据你实际使用的Tokenizer确认
        # 假设 tokenizer 已经包含了 <|im_start|> 等特殊token
        # 如果没有，可以用文本方式拼接，但最好确保 tokenizer 能正确分词
        
        for msg in messages:
            role = msg['role']
            content = msg['content']
            
            # 构建单个消息的文本
            # 注意：这里手动拼接了 ChatML 格式，需确保你的模型支持这种模板
            msg_text = f"<|im_start|>{role}\n{content}<|im_end|>\n"
            
            # 编码当前片段
            msg_ids = self.tokenizer.encode(msg_text, add_special_tokens=False)
            
            input_ids.extend(msg_ids)
            
            if role == 'user':
                #如果是 User 的输入，Label 全部设为 -100 (不计算 Loss)
                labels.extend([self.ignore_index] * len(msg_ids))
            elif role == 'assistant':
                # 如果是 Assistant 的回答，Label 就是 input_ids 本身 (计算 Loss)
                labels.extend(msg_ids)
            else:
                # 系统提示词等其他角色，通常也不计算 Loss
                labels.extend([self.ignore_index] * len(msg_ids))

        # 3. 截断与填充 (Padding)
        # 添加结束符 (可选，视模型而定，通常加上 EOS)
        input_ids.append(self.tokenizer.eos_token_id)
        labels.append(self.tokenizer.eos_token_id)

        # 截断
        if len(input_ids) > self.max_seq_len:
            input_ids = input_ids[:self.max_seq_len]
            labels = labels[:self.max_seq_len]
        
        # 填充 (Padding)
        pad_len = self.max_seq_len - len(input_ids)
        if pad_len > 0:
            # input_ids 填充 pad_token_id (通常是0)
            input_ids = input_ids + [self.tokenizer.pad_token_id] * pad_len
            # labels 填充 -100 (忽略)
            labels = labels + [self.ignore_index] * pad_len

        # 4. 转为 Tensor
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        labels = torch.tensor(labels, dtype=torch.long)

        # 注意：使用 HuggingFace Trainer + AutoModelForCausalLM 时
        # 模型内部会自动进行 shift (错位)，所以 input_ids 和 labels 长度应保持一致
        # 不需要像预训练代码那样手动 X=[:-1], Y=[1:]
        
        return {
            'input_ids': input_ids,
            'labels': labels,
            'attention_mask': input_ids.ne(self.tokenizer.pad_token_id) # 这是一个好习惯，生成 attention mask
        }

    def _process_text_only(self, text):
        """处理只有 text 字段的情况 (降级为类似预训练的处理)"""
        # 简单加上 eos
        text = text + self.tokenizer.eos_token
        input_ids = self.tokenizer.encode(text)
        
        if len(input_ids) > self.max_seq_len:
            input_ids = input_ids[:self.max_seq_len]
        else:
            pad_len = self.max_seq_len - len(input_ids)
            input_ids = input_ids + [self.tokenizer.pad_token_id] * pad_len
            
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        # 对于纯文本，通常 label = input_ids (全量预测)
        labels = input_ids.clone() 
        labels[input_ids == self.tokenizer.pad_token_id] = self.ignore_index
        
        return {
            'input_ids': input_ids,
            'labels': labels,
            'attention_mask': input_ids.ne(self.tokenizer.pad_token_id)
        }
    
# # 内存不够，可使用如下方法加载数据
# class LLMDataset(IterableDataset):
#     def __init__(self, data_path, tokenizer, max_seq_len):
#         super().__init__()
#         self.data_path = data_path
#         self.tokenizer = tokenizer
#         self.max_seq_len = max_seq_len
    
#     def __iter__(self):
#         return self.data_process()
    
#     def data_process(self):
#         with open(self.data_path, 'r', encoding='utf-8') as f:
#             for line in f:
#                 line = json.loads(line)
#                 text = '<s>' + line['text'] + '</s>'
#                 input_ids = self.tokenizer.encode(text)
#                 text_len = len(input_ids)
#                 if text_len > self.max_seq_len:
#                     input_ids = input_ids[:self.max_seq_len]
#                 else:
#                     input_ids = input_ids + [0] * (self.max_seq_len - text_len)
#                 input_ids = np.array(input_ids)
#                 X = np.array(input_ids[:-1]).astype(np.int64)
#                 Y = np.array(input_ids[1:]).astype(np.int64)
#                 yield {
#                     'input_ids': torch.from_numpy(X),
#                     'labels': torch.from_numpy(Y),
                # }

# DPO数据集
class DPODataset(Dataset):
    def __init__(self, data_path, tokenizer):
        """
        DPO数据集初始化.
        :param data_path: 数据路径.
        :param tokenizer: 分词器.
        """
        super().__init__()
        self.data_path = data_path
        self.tokenizer = tokenizer
        
        with open(self.data_path, 'r', encoding='utf-8') as f:
            self.datas = json.load(f)
        
    def __getitem__(self, index):
        """
        获取单个样本.
        :param index: 样本索引.
        :return: 样本.
        """
        sample = self.datas[index]
        prompt = sample['prompt']
        chosen = sample['chosen']
        rejected = sample['rejected']
        messages = [
            {"role": "user", "content": prompt}
        ]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        prompt_inputs = self.tokenizer(text=text)['input_ids']
        rejected_inputs = self.tokenizer(text=rejected)['input_ids'] + [self.tokenizer.eos_token_id]
        chosen_inputs = self.tokenizer(text=chosen)['input_ids'] + [self.tokenizer.eos_token_id]
        return [prompt_inputs, chosen_inputs, rejected_inputs]
    
    def __len__(self):
        """
        返回数据集长度.
        """
        return len(self.datas)
    
    
# DPO数据收集器
class DPODataCollator:
    def __init__(self, tokenizer, max_seq_len):
        """
        DPO数据收集器初始化.
        :param tokenizer: 分词器.
        :param max_seq_len: 最大序列长度.
        """
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
    def __call__(self, features):
        """
        数据收集器.
        :param features: 样本列表.
        :return: 批处理数据.
        """
        inputs_ids = []
        labels = []
        
        for feature in features:
            inputs_ids.append(feature[0] + feature[1])
            labels.append([0]*len(feature[0]) + feature[1])
        for feature in features:
            inputs_ids.append(feature[0] + feature[2])
            labels.append([0]*len(feature[0]) + feature[2])
            
        def process(inputs_ids, labels):
            inputs_ids = [input_ids[:self.max_seq_len] for input_ids in inputs_ids]
            labels = [label[:self.max_seq_len] for label in labels]
            max_len = max([len(input_ids) for input_ids in inputs_ids])
            batch_input_ids = []
            batch_labels = []
            
            for input_ids, label in zip(inputs_ids, labels):
                if len(input_ids) <= max_len:
                    input_ids = input_ids+[0]*(max_len-len(input_ids))
                    label = label+[0]*(max_len-len(label))
                    batch_input_ids.append(input_ids[:-1])
                    batch_labels.append(label[1:])
            return batch_input_ids, batch_labels
        
        inputs_ids, labels = process(inputs_ids, labels)
        
        return {
            "input_ids": torch.tensor(inputs_ids),
            "labels": torch.tensor(labels)
            }




