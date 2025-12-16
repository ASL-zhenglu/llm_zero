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


# RMSNorm层，用于层归一化
class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        RMSNorm 初始化.
        :param hidden_size: 隐藏层大小.
        :param eps: 一个小的数，防止除以零.
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        """
        前向传播.
        :param hidden_states: 输入的隐藏状态.
        :return: 归一化后的隐藏状态.
        """
        hidden_states = hidden_states.float()
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.float()
    
# 旋转位置编码的辅助函数，将输入张量在最后一个维度上分成两半，并交换它们的位置
def rotate_half(x):
    """
    将输入张量在最后一个维度上分成两半，并交换它们的位置.
    :param x: 输入张量.
    :return: 旋转后的张量.
    """
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

# 应用旋转位置编码
def apply_rotate_pos_emb(q, k, cos, sin, unsqueeze_dim=2):
    """
    应用旋转位置编码.
    :param q: 查询向量.
    :param k: 键向量.
    :param cos: 旋转矩阵的余弦部分.
    :param sin: 旋转矩阵的正弦部分.
    :param unsqueeze_dim: 扩展维度的位置.
    :return: 编码后的查询和键向量.
    """
    cos = cos.unsqueeze(unsqueeze_dim) # 扩充第三维度,多头注意力，因为此时（b, Seq_Len, num_heads, head_dim）
    sin = sin.unsqueeze(unsqueeze_dim) # 扩充第三维度,多头注意力，因为此时（b, Seq_Len, num_heads, head_dim）
   
    q_embed = (q*cos) + (rotate_half(q)*sin) # [xcos, ycos] + [-ysin, xsin]
    k_embed = (k*cos) + (rotate_half(k)*sin) # [xcos, ycos] + [-ysin, xsin]
    
    return q_embed, k_embed

# 旋转位置编码模块
class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len=1024):
        """
        旋转位置编码初始化.
        :param dim: 编码维度.
        :param max_seq_len: 最大序列长度.
        """
        super(RotaryEmbedding, self).__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim)) # 
        t = torch.arange(max_seq_len).float().unsqueeze(1) # 有多少词汇，最大序列长度
        freqs = t @ inv_freq.unsqueeze(0) # 计算频率矩阵
        freqs = torch.cat((freqs, freqs), dim=-1) # 复制频率矩阵
        
        self.register_buffer("cos_cached", freqs.cos()) # 余弦缓存
        self.register_buffer("sin_cached", freqs.sin()) # 正弦缓存
        
    def forward(self, q, k):
        """
        前向传播.
        :param q: 查询向量.
        :param k: 键向量.
        :return: 编码后的查询和键向量.
        """
        cos = self.cos_cached[:q.shape[1], :].unsqueeze(0) # 根据查询长度获取对应的余弦值
        sin = self.sin_cached[:q.shape[1], :].unsqueeze(0) # 根据查询长度获取对应的正弦值
        return apply_rotate_pos_emb(q, k, cos, sin)
    
# 重复键值对，用于分组查询注意力
def repeat_kv(hidden_states, n_rep): # [A, B] -> [A, A, B, B]
    """
    重复键值对. q = 16头时，k,v各自从8头变成16头 head_dim不变, 就是d_model/num_heads
    :param hidden_states: 输入的隐藏状态.
    :param n_rep: 重复次数.
    :return: 重复后的隐藏状态.
    """
    batch, slen, num_key_value_heads, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, :, None, :].expand(batch, slen, num_key_value_heads, n_rep, head_dim)
    return hidden_states.reshape(batch, slen, num_key_value_heads * n_rep, head_dim)

# 注意力模块
class Attention(nn.Module):
    def __init__(self, config):
        """
        注意力模块初始化.
        :param config: 模型配置.
        """
        super().__init__()
        self.config = config
        self.dropout = config.dropout
        self.hidden_size = config.hidden_size  # hidden_size，单词特征维度
        self.num_heads = config.num_attention_heads # 注意力头数
        self.head_dim = getattr(config, "head_dim", self.hidden_size // self.num_heads) # 每个头的维度
        self.num_key_value_heads = config.num_key_value_heads # kv头数
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads   # 组数
        self.k_cache, self.v_cache = None, None
        self.is_causal = True # 因为是解码器，所以是因果注意力
        self.flash_attn = self.config.flash_attn # 是否使用flash attention
        
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias) # 线性层只对最后一维进行操作
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=config.attention_bias) # 输出线性变换
        self.residual_dropout = nn.Dropout(self.dropout)
        self.attention_dropout = nn.Dropout(self.dropout)
        self.rotary_emb = RotaryEmbedding(self.head_dim) # 32
        
    def forward(self, hidden_states, use_kv_cache=False):
        """
        前向传播.
        :param hidden_states: 输入的隐藏状态.
        :param use_kv_cache: 是否使用键值缓存.
        :return: 注意力输出.
        """
        b, s = hidden_states.shape[:2]
        if use_kv_cache and self.eval():
            if self.k_cache is None or self.k_cache.shape[1] != s-1:
                # 第一次直接产生q,k,v
                q, k, v = self.q_proj(hidden_states), self.k_proj(hidden_states), self.v_proj(hidden_states)
            else:
                # 来了一个新词，要将它的q，k，v拼接在历史上
                token = hidden_states[:, -1:, :]
                q = torch.cat((torch.zeros_like(hidden_states[:, :-1, :]), self.q_proj(token)), dim=1) # 为了保证RoPetary Embedding的位置对应，所以补0
                # (B, S-1, 512) + (B, 1, 512) -> (B, S, 512)
                k = torch.cat((self.k_cache, self.k_proj(token)), dim=1) # 拼接过去的k
                v = torch.cat((self.v_cache, self.v_proj(token)), dim=1) # 拼接过去的v
            self.k_cache, self.v_cache = k, v
            
        else:
            q, k, v = self.q_proj(hidden_states), self.k_proj(hidden_states), self.v_proj(hidden_states)
            
        q = q.view(b, s, self.num_heads, self.head_dim) # [bacth, 512, 16, 32]
        k = k.view(b, s, self.num_key_value_heads, self.head_dim) # [bacth, 512, 8, 32]
        v = v.view(b, s, self.num_key_value_heads, self.head_dim) # [bacth, 512, 8, 32]
        
        q, k = self.rotary_emb(q, k) # 旋转位置编码只需要对q, k 进行编码 (B, 512, 8, 32)
        
        k = repeat_kv(k, self.num_key_value_groups) # k,v从8头变成16头, num_key_value_groups = 2
        v = repeat_kv(v, self.num_key_value_groups) # k,v从8头变成16头
        
        q = q.transpose(1, 2) # b, self.num_heads, s, self.head_dim
        k = k.transpose(1, 2) # b, self.num_heads, s, self.head_dim
        v = v.transpose(1, 2) # b, self.num_heads, s, self.head_dim
        
        if self.flash_attn:
            
            # q*k转置，（b, self.num_heads, s, self.head_dim）* (b, self.num_heads, self.head_dim，s) = （b, self.num_heads, s, s）
            # q*k/sqrt(self.head_dim)*v  （b, self.num_heads, s, s）* (b, self.num_heads, s, self.head_dim) = b, self.num_heads, s, self.head_dim
            output = F.scaled_dot_product_attention(q, k, v, attn_mask=None, 
                                                    dropout_p=self.dropout if self.training else 0.0, 
                                                    is_causal=self.is_causal) 
        else:
            mask = torch.full((1, 1, self.config.max_seq_len, self.config.max_seq_len), float("-inf"))  # 初始化掩码
            mask = torch.triu(mask, diagonal=1)  # 生成上三角掩码
            scores = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(self.head_dim)  # 计算注意力分数
            scores = scores + self.mask[:, :, :s, :s]  # 应用掩码
            scores = F.softmax(scores.float(), dim=-1).type_as(q)  # 计算 softmax, type_as 保持数据类型一致
            scores = self.attention_dropout(scores)  # 应用注意力 dropout
            output = torch.matmul(scores, v)  # 计算输出
        
        output = output.transpose(1, 2).contiguous().view(b, s, -1) # b, s, self.hidden_size
        
        output = self.o_proj(output)
        output = self.residual_dropout(output)
        return output
    
    
# MLP模块
class MLP(nn.Module):
    def __init__(self, config):
        """
        MLP模块初始化.
        :param config: 模型配置.
        """
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size # 512
        self.intermediate_size = config.intermediate_size # 2048
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=config.mlp_bias)
        
    def forward(self, x):
        """
        前向传播.
        :param x: 输入张量.
        :return: MLP输出.
        """
        down_proj = self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
        return down_proj
    
# 解码器层
class DecoderLayer(nn.Module):
    def __init__(self, config, layer_idx):
        """
        解码器层初始化.
        :param config: 模型配置.
        :param layer_idx: 层索引.
        """
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = Attention(config)
        self.mlp = MLP(config)
        self.input_layernorm = RMSNorm(config.hidden_size)
        self.post_attention_layernorm = RMSNorm(config.hidden_size)
        self.layer_idx = layer_idx # 当前层的索引，用于调试或其他目的
    def forward(
        self,
        hidden_states,
        use_kv_cache
    ):
        """
        前向传播.
        :param hidden_states: 输入的隐藏状态.
        :param use_kv_cache: 是否使用键值缓存.
        :return: 解码器层输出.
        """
        residual = hidden_states # 当前层的输入作为残差连接

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            use_kv_cache=use_kv_cache
        )
        
        hidden_states = residual + hidden_states
        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        outputs = residual + hidden_states
        return outputs
   
   
# 编写自定义配置时需要记住的三个重要事项如下：
# 1、必须继承自 PretrainedConfig
# 2、PretrainedConfig 的 __init__ 方法必须接受任何 kwargs
# 3、这些 kwargs 需要传递给超类的 __init__ 方法。
class Config(PretrainedConfig):
    model_type = "small_model"
    
    def __init__(self,
                hidden_size = 512, 
                num_attention_heads = 16,
                num_key_value_heads = 8,
                flash_attn = True,
                attention_bias = False,
                max_seq_len = 512, 
                intermediate_size = 2048,
                mlp_bias = False,
                vocab_size = 6400,
                n_layers = 8,
                dropout = 0.0,
                **kwargs):
        """
        模型配置初始化.
        :param hidden_size: 隐藏层大小.
        :param num_attention_heads: 注意力头数.
        :param num_key_value_heads: 键值对头数.
        :param flash_attn: 是否使用flash attention.
        :param attention_bias: 注意力是否有偏置.
        :param max_seq_len: 最大序列长度.
        :param intermediate_size: MLP中间层大小.
        :param mlp_bias: MLP是否有偏置.
        :param vocab_size: 词表大小.
        :param n_layers: 解码器层数.
        :param dropout: dropout概率.
        :param kwargs: 其他参数.
        """
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.flash_attn = flash_attn
        self.attention_bias = attention_bias
        self.max_seq_len = max_seq_len
        self.intermediate_size = intermediate_size
        self.mlp_bias = mlp_bias
        self.vocab_size = vocab_size
        self.n_layers = n_layers
        self.dropout = dropout
        super().__init__(**kwargs)
         

# LLM模型
class LLM(PreTrainedModel):
    config_class = Config
    
    def __init__(self, config):
        """
        LLM模型初始化.
        :param config: 模型配置.
        """
        super().__init__(config)
        self.config = config
        self.vocab_size = self.config.vocab_size # 词表大小 6400
        self.n_layers = self.config.n_layers # 解码器层数 8

        self.token_embeddings = nn.Embedding(self.config.vocab_size, self.config.hidden_size) # 词嵌入层 [6400, 512]
        self.dropout = nn.Dropout(self.config.dropout) 
        self.layers = torch.nn.ModuleList() 
        for layer_idx in range(self.n_layers):
            self.layers.append(DecoderLayer(self.config, layer_idx)) 
        self.norm = RMSNorm(self.config.hidden_size) 
        self.output = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False) 
        self.token_embeddings.weight = self.output.weight
        self.apply(self._init_weights) 
        self.loss = None 
        
        for pn, p in self.named_parameters():
            if pn.endswith('w3.weight') or pn.endswith('wo.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * self.config.n_layers)) 

    def _init_weights(self, module):
        """
        初始化模型权重.
        :param module: 模型模块.
        """
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)  
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)  
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02) 
            
        
    def forward(self, input_ids, labels, use_kv_cache=False):
        """
        前向传播.
        :param input_ids: 输入token id.
        :param labels: 标签.
        :param use_kv_cache: 是否使用键值缓存.
        :return: 模型输出.
        """
       
        hidden_states = self.token_embeddings(input_ids) # [B, T, hidden_size]
        hidden_states = self.dropout(hidden_states)  
        for idx, layer in enumerate(self.layers):
            hidden_states = layer(hidden_states, use_kv_cache=use_kv_cache)  

        hidden_states = self.norm(hidden_states) 

        if labels is not None: # 说明我们在训练
            logits = self.output(hidden_states)  
            self.loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100) 
            # 函数输出要求是二维的, “如果标签是 -100（通常是 Padding 补全符），就不要计算 Loss，也不要更新梯度。”这保证了模型不会去学习无意义的补全符
        else: # 推理
            logits = self.output(hidden_states[:, [-1], :])
            self.loss = None  

        return CausalLMOutputWithPast(self.loss, logits)
    
    @torch.inference_mode
    def generate(self, inputs, eos, max_new_tokens, temperature=0.7, top_k=None, stream=True, repetition_penalty=1.,
                 use_kv_cache=True):
        """
        生成文本.
        :param inputs: 输入.
        :param eos: 结束符.
        :param max_new_tokens: 最大生成token数.
        :param temperature: 温度系数.
        :param top_k: top-k采样.
        :param stream: 是否流式生成.
        :param repetition_penalty: 重复惩罚.
        :param use_kv_cache: 是否使用键值缓存.
        :return: 生成的token.
        """
        
        input_ids = inputs['input_ids']
        labels = inputs['labels']
        s = input_ids.shape[1]
        while input_ids.shape[1] < max_new_tokens - 1:   # 没用达到最大长度
            inference_res = self(input_ids, labels, use_kv_cache=use_kv_cache)   # 使用forword函数进行推理
            logits = inference_res.logits # 取最后一个token的logits(batch_size, seq_len, vocab_size)
            logits = logits[:, -1, :] # 降维度了

            # input_ids.tolist()[0] 把当前已生成的整个序列（Tensor）转成 Python 列表
            # set(...) 去重，只关心出现过哪些词，不关心出现次数
            for token in set(input_ids.tolist()[0]):  # 防止模型变成复读机
                # logits[:, token] 选中这些“出现过的词”对应的预测分数
                # /= repetition_penalty 将这些分数除以惩罚系数
                logits[:, token] /= repetition_penalty
            
            if temperature == 0.0:  # 绝对理性
                _, idx_next = torch.topk(logits, k=1, dim=-1)
            else:
                # 当 T < 1 (低温)：比如 logits / 0.1。原本大的分数变得极大，小的变得极小。分布变得尖锐。模型更倾向于选高概率词，更保守。
                # 当 T > 1 (高温)：比如 logits / 2.0。原本差异很大的分数，差距被缩小了。分布变得平坦。低概率的词也有机会被选中，模型更“疯狂”、更有创造力，但也更容易胡说八道。
                logits = logits / temperature  
                if top_k is not None:  
                    # # 1. 找到第 k 大的分数作为阈值 v
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    # # 2. 把所有小于阈值的分数设为负无穷 (-Inf)
                    logits[logits < v[:, [-1]]] = -float('Inf') 

                probs = F.softmax(logits, dim=-1)  
                # 轮盘赌采样
                idx_next = torch.multinomial(probs, num_samples=1, generator=None)  

            if idx_next == eos:  
                break

            input_ids = torch.cat((input_ids, idx_next), dim=1)  
            if stream:  
                yield input_ids[:, s:]  

        if not stream:  
            yield input_ids[:, s:]  
               
if __name__ == '__main__':   

    # 模型配置
    config = Config()
    # 实例化模型
    model = LLM(config)
    # 打印模型参数量
    print(f'模型参数量为：{sum(p.numel() for p in model.parameters() if p.requires_grad)}')

    # 数据收集器
    data_collator = DefaultDataCollator()
    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained("./tokenizer", use_fast=True)
    # 训练参数
    args = TrainingArguments(output_dir='./results2048', 
                            num_train_epochs=10, 
                            do_train=True, 
                            per_device_train_batch_size=32,
                            gradient_accumulation_steps=8,
                            # max_steps=15000,
                            logging_steps=100, # 每训练一百步记录一次日志
                            report_to='tensorboard',
                            save_total_limit=5, # 最多保存5个检查点
                            bf16=True,  # 使用bf16混合精度训练
                            learning_rate=2e-4,
                            lr_scheduler_type='cosine', # 学习率调用策略
                            dataloader_num_workers=8,  # 数据加载线程数
                            dataloader_pin_memory=True,  # 是否将数据加载到固定内存
                            save_safetensors=False)          
    # 加载数据集
    dataset = LLMDataset('./dataset/pretrain_hq.jsonl', tokenizer=tokenizer, max_seq_len=512)
    # 实例化训练器
    trainer = Trainer(model=model, args=args, train_dataset=dataset, tokenizer=tokenizer, data_collator=data_collator)
    # 如果是初次训练resume_from_checkpoint为false，接着checkpoint继续训练，为True
    trainer.train(resume_from_checkpoint=True)
    # 保存模型
    trainer.save_model('./saves/model')
    # 保存训练状态
    trainer.save_state()
