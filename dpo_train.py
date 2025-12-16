# 导入必要的库
from transformers import TrainingArguments, Trainer, AutoModelForCausalLM, AutoTokenizer, AutoConfig
import torch
import torch.nn.functional as F
from dataset import DPODataset, DPODataCollator
from train import LLM, Config


# 将logits转换为概率
def logits_to_probs(logits, labels):
    """
    将logits转换为概率.
    :param logits: 模型的logits输出.
    :param labels: 真实标签.
    :return: 对应标签的概率.
    """
    # logits shape: (batch_size, seq_len, vocab_size)
    # labels shape: (batch_size, seq_len)
    # probs shape: (batch_size, seq_len)
    log_probs = F.log_softmax(logits, dim=2)
    probs = torch.gather(log_probs, dim=2, index=labels.unsqueeze(2)).squeeze(-1)
    return probs

# 掩码logits
def mask_logits(logits, labels):
    """
    根据标签掩码logits.
    :param logits: 输入的logits.
    :param labels: 标签.
    :return: 掩码后的logits.
    """
    # logits shape: (batch_size, seq_len, vocab_size)
    # labels_masks shape: (batch_size, seq_len)
    new_logits = []
    for logit, label in zip(logits, labels):
        new_logits.append(logit[label != 0].sum().unsqueeze(0))
    
    return new_logits


# DPO损失函数
def dpo_loss(ref_probs, probs, beta):
    """
    DPO损失函数.
    :param ref_probs: 参考模型的概率.
    :param probs: 当前模型的概率.
    :param beta: 温度系数.
    :return: DPO损失.
    """
    def split_probs(probs):
        len_chosen = int(len(probs) // 2)
        chosen_data = probs[:len_chosen]
        reject_data = probs[len_chosen:]
        return torch.cat(chosen_data), torch.cat(reject_data)
    
    ref_chosen_probs, ref_reject_probs = split_probs(ref_probs)
    chosen_probs, reject_probs = split_probs(probs) # 输出对应的log概率
    pi_logratios = chosen_probs - reject_probs 
    ref_logratios = ref_chosen_probs - ref_reject_probs
    logits = pi_logratios - ref_logratios
    loss = -F.logsigmoid(beta*logits)
    return loss.mean()
    


# DPO训练器
class DPOTrainer(Trainer):
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        计算损失.
        :param model: 模型.
        :param inputs: 输入.
        :param return_outputs: 是否返回输出.
        :param num_items_in_batch: 批次中的项目数.
        :return: 损失.
        """
        input_ids = inputs['input_ids'] # (batch_size * 2, seq_len)
        labels = inputs['labels'] # (batch_size * 2, seq_len)
        with torch.no_grad():
            ref_logits = ref_model(input_ids=input_ids, labels = labels).logits
        ref_probs = logits_to_probs(ref_logits, labels)
        ref_probs = mask_logits(ref_probs, labels)
        logits = model(input_ids=input_ids, labels = labels).logits
        probs = logits_to_probs(logits, labels)
        probs = mask_logits(probs, labels)
        loss = dpo_loss(ref_probs, probs, 0.1)
        return loss

    # def training_step(
    #     self, model, inputs, num_items_in_batch=None
    # ) -> torch.Tensor:
    #     input_ids = inputs['input_ids']
    #     labels = inputs['labels']
    #     with torch.no_grad():
    #         ref_logits = ref_model(input_ids=input_ids, labels = labels).logits
    #     ref_probs = logits_to_probs(ref_logits, labels)
    #     ref_probs = mask_logits(ref_probs, labels)
    #     # 因为参考模型的累计概率不发生变化，为了尽量减少多次计算，计算一次参考模型的累积概率，多训练几次需要优化的模型
    #     for _ in range(1):
            
    #         model.train()
    #         logits = model(input_ids=input_ids, labels = labels).logits
    #         probs = logits_to_probs(logits, labels)
    #         probs = mask_logits(probs, labels)
        
    #         if hasattr(self.optimizer, "train") and callable(self.optimizer.train):
    #             self.optimizer.train()

    #         with self.compute_loss_context_manager():
    #             # loss = self.compute_loss(model, inputs, num_items_in_batch=num_items_in_batch)
    #             loss = dpo_loss(ref_probs, probs, 0.2)

    #         # del inputs
    #         if (
    #             self.args.torch_empty_cache_steps is not None
    #             and self.state.global_step % self.args.torch_empty_cache_steps == 0
    #         ):
                
    #             torch.cuda.empty_cache()

    #         kwargs = {}

    #         if self.args.n_gpu > 1:
    #             loss = loss.mean()  # mean() to average on multi-gpu parallel training

    #         self.accelerator.backward(loss, retain_graph=True, **kwargs)
    #     # Finally we need to normalize the loss for reporting
    #     if num_items_in_batch is None:
    #         return loss.detach() / self.args.gradient_accumulation_steps
    #     return loss.detach()
    
        
if __name__ == "__main__":
    # 注册自定义模型和配置
    AutoConfig.register("small_model", Config)
    AutoModelForCausalLM.register(Config, LLM)
    # 加载SFT后的模型
    model = AutoModelForCausalLM.from_pretrained('/home/user/wyf/train_model_from_scratch/saves/sft')

    print(f'模型可训练参数量为：{sum(p.numel() for p in model.parameters() if p.requires_grad)}')
    # 加载参考模型
    ref_model = AutoModelForCausalLM.from_pretrained('/home/user/wyf/train_model_from_scratch/saves/sft').eval().to('cuda')
    
    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained("/home/user/wyf/train_model_from_scratch/tokenizer", use_fast=True)
    # DPO数据收集器
    data_collator = DPODataCollator(tokenizer, max_seq_len=512) # 加载的大模型旋转位置编码最大长度为1024，这里不能超过这个值
    # 训练参数
    args = TrainingArguments(output_dir='./dpo-1-epoch', 
                            num_train_epochs=1,  # 训练太多轮，模型似乎会输出很多重复内容
                            do_train=True, 
                            per_device_train_batch_size=16,
                            gradient_accumulation_steps=4,
                            # max_steps=15000,
                            logging_steps=50,
                            report_to='tensorboard',
                            save_total_limit=3,
                            bf16=True,
                            learning_rate=0.00001,  # 学习率很重要，太大会把模型训飞
                            lr_scheduler_type='cosine',
                            dataloader_num_workers=1,
                            dataloader_pin_memory=True,
                            save_safetensors=False,
                            save_steps=100)          
    # 加载DPO数据集
    dataset = DPODataset('/home/user/wyf/train_model_from_scratch/dataset/dpo_data_512.json', tokenizer=tokenizer)
    # 实例化DPO训练器
    trainer = DPOTrainer(model=model, args=args, train_dataset=dataset, tokenizer=tokenizer, data_collator=data_collator)
    
    # 如果是初次训练resume_from_checkpoint为false，接着checkpoint继续训练，为True
    trainer.train(resume_from_checkpoint=False)
    # 保存模型
    trainer.save_model('./saves/dpo-1-epoch')
    # 保存训练状态
    trainer.save_state()