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
from dataset import SFTDataset, LLMDataset
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import torch
from train import LLM, Config

if __name__ == '__main__':
    # 注册自定义模型和配置
    AutoConfig.register("small_model", Config)
    AutoModelForCausalLM.register(Config, LLM)
    # 加载预训练模型
    model = AutoModelForCausalLM.from_pretrained('./saves/model')
    # 打印模型参数量
    print(f'模型参数量为：{sum(p.numel() for p in model.parameters() if p.requires_grad)}')

    # 数据收集器
    data_collator = DefaultDataCollator()
    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained("./tokenizer", use_fast=True)
    # 检查并设置 pad_token，如果不存在则使用 eos_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print(f"pad_token is None, set to eos_token: {tokenizer.pad_token}")
    # 训练参数
    args = TrainingArguments(output_dir='./sft', 
                            num_train_epochs=5, 
                            do_train=True, 
                            per_device_train_batch_size=8,
                            gradient_accumulation_steps=8,
                            # max_steps=15000,
                            logging_steps=100,
                            report_to='tensorboard',
                            save_total_limit=5,
                            bf16=True,
                            learning_rate=2e-4,
                            lr_scheduler_type='cosine',
                            dataloader_num_workers=1,
                            dataloader_pin_memory=True,
                            save_safetensors=False)          
    # 加载SFT数据集
    dataset = SFTDataset('./dataset/sft_mini_512.jsonl', tokenizer=tokenizer, max_seq_len=1024)
    # 实例化训练器
    trainer = Trainer(model=model, args=args, train_dataset=dataset, tokenizer=tokenizer, data_collator=data_collator)
    # 如果是初次训练resume_from_checkpoint为false，接着checkpoint继续训练，为True
    trainer.train(resume_from_checkpoint=True)
    # 保存模型
    trainer.save_model('./saves/sft')
    # 保存训练状态
    trainer.save_state()
