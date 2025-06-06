from transformers import AutoModelForCausalLM,AutoModel,AutoModelForSequenceClassification,AutoTokenizer,PreTrainedModel000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
from dataclasses import dataclass
import random
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
from torch.utils.tensorboard import SummaryWriter
from typing import Callable,Dict,List,Optional,Tuple,Union,Any
from copy import deepcopy
from datasets import load_dataset
from reward_func import *
import os

class GSM8KDataset(Dataset):
    def __init__(self,data_path,tokenizer):
        
        self.tokenizer = tokenizer
        data = load_dataset(data_path)
        self.data = data['train']
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        sample = self.data[index]
        answer = sample['answer_only']
        prompt = sample['question_zh-cn']
        return {'prompy':prompt,'answer':answer}
    
@dataclass
class Samples:
    prompt_response_ids:torch.Tensor
    response_ids:torch.Tensor
    prompt:Any
    answer:Any
    attention_mask:Optional[torch.LongTensor]
    action_mask:Optional[torch.BoolTensor]
    num_actions:Union[int,torch.Tensor]
    response_length:int

class GRPOArguments:
    
    output_dir = './output'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    lr = 0.00001
    save_steps = 100
    epoch = 3
    num_generations = 4 # 组内样本数
    max_prompt_length = 256 # 最大输出长度
    reward_weights : List[float] = None # 奖励函数的权重
    beta = 0.0 # KL散度系数，为0时则忽略KL散度，不是用参考模型
    clip_eps = 0.2
    gradient_accumulation_steps = 2 #梯度累加，积累两次梯度后才更新一次参数
    num_iterations = 1 # 采样一次样本训练模型轮数
    batch_size = 1
    
class GRPOTrainer:
    def __init__(self,
                 model = None,
                 reward_funcs:Union[List[str],List[Callable]] = None,
                 args = None,
                 train_dataset:Optional[Dataset] = None,
                 eval_dataset:Optional[Dataset] = None,
                 tokenizer = None,
                 reward_tokenizer = None):
        
        self.args = args
        # 加载模型
        if isinstance(model,str): #如果model参数是字符串类型，就加载模型
            model = AutoModelForCausalLM.from_pretrained(model)
        self.model = model.to(self.args.device)
        
        # 是否使用参考模型
        self.ref_model = None
        if self.args.beta != 0.0:
            self.ref_model = deepcopy()
            self.ref_model.eval()
            
        if isinstance(tokenizer,str):
            tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        self.tokenizer = self.get_tokenizer(tokenizer)
        
        if isin
    