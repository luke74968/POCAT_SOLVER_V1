# utils/functions.py

import math
import torch
from torch import Tensor
from tensordict import TensorDict
from typing import Union

def _batchify_single(x: Union[Tensor, TensorDict], repeats: int) -> Union[Tensor, TensorDict]:
    """텐서 또는 TensorDict의 0번 차원(배치)을 repeats 만큼 복제하여 확장합니다."""
    s = x.shape
    return x.expand(repeats, *s).contiguous().view(s[0] * repeats, *s[1:])

def batchify(x: Union[Tensor, TensorDict], shape: Union[tuple, int]) -> Union[Tensor, TensorDict]:
    """
    POMO와 같이 여러 개의 시작점을 사용할 때 데이터를 효율적으로 확장합니다.
    예: x.shape: [a, ...], shape: [b] -> out.shape: [a*b, ...]
    """
    shape = [shape] if isinstance(shape, int) else shape
    for s in reversed(shape):
        x = _batchify_single(x, s) if s > 0 else x
    return x

def _unbatchify_single(x: Union[Tensor, TensorDict], repeats: int) -> Union[Tensor, TensorDict]:
    """batchify된 데이터를 원래 형태로 되돌립니다."""
    s = x.shape
    return x.view(repeats, s[0] // repeats, *s[1:]).permute(1, 0, *range(2, len(s) + 1))

def unbatchify(x: Union[Tensor, TensorDict], shape: Union[tuple, int]) -> Union[Tensor, TensorDict]:
    """
    batchify된 데이터를 원래 형태로 되돌립니다.
    예: x.shape: [a*b, ...], shape: [b] -> out.shape: [a, b, ...]
    """
    shape = [shape] if isinstance(shape, int) else shape
    for s in reversed(shape):
        x = _unbatchify_single(x, s) if s > 0 else x
    return x

def gather_by_index(src: Tensor, idx: Tensor, dim: int = 1, squeeze: bool = True) -> Tensor:
    """
    주어진 인덱스(idx)에 따라 소스 텐서(src)에서 값을 추출합니다.
    디코더에서 특정 노드의 임베딩을 가져오는 등 자주 사용됩니다.
    """
    expanded_shape = list(src.shape)
    expanded_shape[dim] = -1
    idx = idx.view(idx.shape + (1,) * (src.dim() - idx.dim())).expand(expanded_shape)
    squeeze = idx.size(dim) == 1 and squeeze
    return src.gather(dim, idx).squeeze(dim) if squeeze else src.gather(dim, idx)

def clip_grad_norms(param_groups, max_norm=math.inf):
    """
    모델의 그래디언트가 폭발하는 것을 방지하기 위해 일정 크기 이상으로 커지지 않도록 잘라냅니다.
    안정적인 훈련을 위한 필수적인 기술입니다.
    """
    grad_norms = [
        torch.nn.utils.clip_grad_norm_(
            group['params'],
            max_norm if max_norm > 0 else math.inf,
            norm_type=2
        )
        for group in param_groups
    ]
    grad_norms_clipped = [min(g_norm, max_norm) for g_norm in grad_norms] if max_norm > 0 else grad_norms
    return grad_norms, grad_norms_clipped