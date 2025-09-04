# pocat_utils.py
import time
import sys
import os
import logging
import math
import torch
from torch import Tensor
from tensordict import TensorDict
from typing import Union

#################################################################
# Logging and Timing Utilities
#################################################################

class AverageMeter:
    """ 여러 값의 평균을 계속 추적하는 클래스 """
    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.sum += (val * n)
        self.count += n

    @property
    def avg(self):
        return self.sum / self.count if self.count else 0

class TimeEstimator:
    """ 훈련 시간 및 남은 시간을 예측하는 클래스 """
    def __init__(self, logger=None):
        self.logger = logger if logger else logging.getLogger('TimeEstimator')
        self.start_time = time.time()
        self.count_zero = 0

    def reset(self, count=1):
        self.start_time = time.time()
        self.count_zero = count - 1

    def get_est(self, count, total):
        curr_time = time.time()
        elapsed_time = curr_time - self.start_time
        remain = total - count
        if count - self.count_zero == 0: # 0으로 나누기 방지
            return elapsed_time / 3600.0, 0.0
            
        remain_time = elapsed_time * remain / (count - self.count_zero)
        return elapsed_time / 3600.0, remain_time / 3600.0

    def get_est_string(self, count, total):
        elapsed_time_h, remain_time_h = self.get_est(count, total)

        elapsed_time_str = f"{elapsed_time_h:.2f}h" if elapsed_time_h > 1.0 else f"{elapsed_time_h*60:.2f}m"
        remain_time_str = f"{remain_time_h:.2f}h" if remain_time_h > 1.0 else f"{remain_time_h*60:.2f}m"

        return elapsed_time_str, remain_time_str

    def print_est_time(self, count, total):
        elapsed_str, remain_str = self.get_est_string(count, total)
        self.logger.info(
            f"Epoch {count:3d}/{total:3d}: Time Est.: Elapsed[{elapsed_str}], Remain[{remain_str}]"
        )

#################################################################
# Tensor Manipulation Utilities
#################################################################

def _batchify_single(x: Union[Tensor, TensorDict], repeats: int) -> Union[Tensor, TensorDict]:
    """ 텐서 또는 TensorDict의 첫 번째 차원(배치)을 'repeats'만큼 복제합니다. """
    s = x.shape
    return x.expand(repeats, *s).contiguous().view(s[0] * repeats, *s[1:])

def batchify(x: Union[Tensor, TensorDict], repeats: int) -> Union[Tensor, TensorDict]:
    """ POMO 스타일의 병렬 탐색을 위해 데이터를 확장하는 함수. """
    return _batchify_single(x, repeats) if repeats > 0 else x

def _unbatchify_single(x: Union[Tensor, TensorDict], repeats: int) -> Union[Tensor, TensorDict]:
    """ batchify의 역연산. """
    s = x.shape
    return x.view(repeats, s[0] // repeats, *s[1:]).permute(1, 0, *range(2, len(s) + 1))

def unbatchify(x: Union[Tensor, TensorDict], repeats: int) -> Union[Tensor, TensorDict]:
    """ POMO 결과를 합치기 위해 확장된 데이터를 원래 형태로 되돌리는 함수. """
    return _unbatchify_single(x, repeats) if repeats > 0 else x

#################################################################
# Training Utilities
#################################################################

def clip_grad_norms(param_groups, max_norm=math.inf):
    """ PyTorch 옵티마이저의 파라미터 그룹에 대해 그래디언트 클리핑을 수행합니다. """
    grad_norms = [
        torch.nn.utils.clip_grad_norm_(
            group['params'],
            max_norm if max_norm > 0 else math.inf,
            norm_type=2
        )
        for group in param_groups if group['params']
    ]
    grad_norms_cpu = [g.item() for g in grad_norms]
    grad_norms_clipped = [min(g_norm, max_norm) for g_norm in grad_norms_cpu] if max_norm > 0 else grad_norms_cpu
    return grad_norms_cpu, grad_norms_clipped
