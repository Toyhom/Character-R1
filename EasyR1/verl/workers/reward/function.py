# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import importlib.util
import os
import sys
from abc import ABC, abstractmethod
from collections import defaultdict
from functools import partial
from typing import Callable, Dict, List, Optional, Tuple, TypedDict

import torch
from transformers import PreTrainedTokenizer

from ...protocol import DataProto
from .config import RewardConfig


class RewardScore(TypedDict):
    overall: float
    format: Optional[float]
    accuracy: Optional[float]


SequentialRewardFunction = Callable[[str|dict, str], RewardScore]

BatchRewardFunction = Callable[[List[str|dict], List[str]], List[RewardScore]]


class FunctionRewardManager(ABC):
    """Reward manager for rule-based reward."""

    def __init__(self, config: RewardConfig, tokenizer: PreTrainedTokenizer):
        if config.reward_function is None:
            raise ValueError("Reward function is not provided.")

        if not os.path.exists(config.reward_function):
            raise FileNotFoundError(f"Reward function file {config.reward_function} not found.")

        spec = importlib.util.spec_from_file_location("custom_reward_fn", config.reward_function)
        module = importlib.util.module_from_spec(spec)
        try:
            sys.modules["custom_reward_fn"] = module
            spec.loader.exec_module(module)
        except Exception as e:
            raise RuntimeError(f"Failed to load reward function: {e}")

        if not hasattr(module, config.reward_function_name):
            raise AttributeError(f"Module {module} does not have function {config.reward_function_name}.")

        reward_fn = getattr(module, config.reward_function_name)
        print(f"Using reward function `{config.reward_function_name}` from `{config.reward_function}`.")
        self.reward_fn = partial(reward_fn, **config.reward_function_kwargs)
        self.config = config
        self.tokenizer = tokenizer

    @abstractmethod
    def compute_reward(self, data: DataProto) -> Tuple[torch.Tensor, Dict[str, List[float]]]:
        """Compute reward for a batch of data."""
        ...


class SequentialFunctionRewardManager(FunctionRewardManager):
    reward_fn: SequentialRewardFunction

    def compute_reward(self, data: DataProto) -> Tuple[torch.Tensor, Dict[str, List[float]]]:
        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_metrics = defaultdict(list)
        response_ids = data.batch["responses"]
        response_length = data.batch["response_mask"].sum(dim=-1)
        for i in range(len(data)):
            valid_response_ids = response_ids[i][: response_length[i]]
            response_str = self.tokenizer.decode(
                valid_response_ids, skip_special_tokens=self.config.skip_special_tokens
            )
            ground_truth = data.non_tensor_batch["ground_truth"][i]

            score = self.reward_fn(response_str, ground_truth)
            reward_tensor[i, response_length[i] - 1] = score["overall"]
            for key, value in score.items():
                reward_metrics[key].append(value)

        return reward_tensor, reward_metrics


class BatchFunctionRewardManager(FunctionRewardManager):
    reward_fn: BatchRewardFunction

    def compute_reward(self, data: DataProto) -> Tuple[torch.Tensor, Dict[str, List[float]]]:

        """
        返回:
            - reward_tensors_dict (Dict[str, torch.Tensor]): 
              一个字典，键是奖励名称 (e.g., "format", "bleu1")，
              值是对应的 token 级奖励张量 (shape: [batch_size, seq_len])。
            - reward_metrics (Dict[str, List[float]]): 
              用于日志记录的原始分数列表。
        """

        response_str, ground_truth, prompts = [], [], []
        response_ids = data.batch["responses"]
        response_length = data.batch["response_mask"].sum(dim=-1)
        # valid_response_ids_list, ground_truth_ids_list, raw_prompt_ids_list = [], [], []

        for i in range(len(data)):
            valid_response_ids = response_ids[i][: response_length[i]]
            response_str.append(
                self.tokenizer.decode(valid_response_ids, skip_special_tokens=self.config.skip_special_tokens)
            )
            ground_truth.append(data.non_tensor_batch["ground_truth"][i])
            prompts.append(data.non_tensor_batch["prompt"][i])

            # valid_response_ids_list.append(valid_response_ids)
            # ground_truth_ids_list.append(data.tensor_batch["ground_truth_ids"][i][:data.non_tensor_batch["ground_truth_ids_length"][i]].tolist())
            # raw_prompt_ids_list.append(data.non_tensor_batch["raw_prompt_ids"][i][:data.non_tensor_batch["raw_prompt_ids_length"][i]].tolist())

        # 这一步调用你的核心奖励计算函数，返回一个分数词典的列表
        # scores: [{'format': 1.0, 'bleu1': 0.8, ...}, {'format': 0.0, 'bleu1': 0.6, ...}]
        scores_list = self.reward_fn(response_str, ground_truth, prompts)

        # 初始化用于日志的 reward_metrics
        reward_metrics = defaultdict(list)
        
        # 初始化用于 PPO 训练的 reward_tensors_dict
        reward_tensors_dict = {}
        if not scores_list:
            return reward_tensors_dict, dict(reward_metrics)

        # 1. 将分数词典列表转换为一个词典，其值为列表
        # scores_by_name: {'format': [1.0, 0.0], 'bleu1': [0.8, 0.6], ...}
        scores_by_name = defaultdict(list)
        for score_dict in scores_list:
            for key, value in score_dict.items():
                # # "overall" 是加权和，我们不再需要它，只处理独立的奖励分量
                # if key != "response_length":
                scores_by_name[key].append(value)

        # 2. 为每个奖励分量创建 token 级别的奖励张量
        device = response_ids.device
        batch_size = len(data)
        batch_indices = torch.arange(batch_size, device=device)
        # 找到每个序列最后一个 token 的索引
        last_token_indices = (response_length - 1).clamp(min=0)

        for name, values in scores_by_name.items():
            # 用于日志记录
            reward_metrics[name].extend(values)

            # 创建一个全零张量
            reward_tensor = torch.zeros_like(response_ids, dtype=torch.float32, device=device)
            
            # 将这批奖励分数一次性地放置在每个序列的最后一个 token 位置
            # tensor_values = torch.tensor(values, dtype=torch.float32, device=device)

            for i in range(batch_size):
                reward_tensor[i, last_token_indices[i]] = values[i]
            
            # 存入字典
            reward_tensors_dict[name] = reward_tensor

        return reward_tensors_dict, dict(reward_metrics)
