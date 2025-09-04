# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
from tensordict import TensorDict

# 직접 정의한 유틸리티와 정의 파일을 임포트
from pocat_defs import FEATURE_DIM
from pocat_utils import batchify
from pocat_env import PocatEnv


class RMSNorm(nn.Module):
    """ LLaMA에서 사용하는 RMSNorm """
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight
    
class Normalization(nn.Module):
    """ Add & Norm 또는 LayerNorm 등을 선택적으로 사용하는 클래스 """
    def __init__(self, embedding_dim, norm_type='rms', **kwargs):
        super().__init__()
        self.norm_type = norm_type
        if self.norm_type == 'layer':
            self.norm = nn.LayerNorm(embedding_dim)
        elif self.norm_type == 'rms':
            self.norm = RMSNorm(embedding_dim)
        elif self.norm_type == 'instance':
            self.norm = nn.InstanceNorm1d(embedding_dim, affine=True, track_running_stats=False)
        else:
            raise NotImplementedError

    def forward(self, x):
        if self.norm_type == 'instance':
            return self.norm(x.transpose(1, 2)).transpose(1, 2)
        else:
            return self.norm(x)
        
class ParallelGatedMLP(nn.Module):
    """ StripedHyena에서 사용하는 FFN의 한 종류 """
    def __init__(self, hidden_size: int, **kwargs):
        super().__init__()
        inner_size = int(2 * hidden_size * 4 / 3)
        multiple_of = 256
        inner_size = multiple_of * ((inner_size + multiple_of - 1) // multiple_of)

        self.l1 = nn.Linear(hidden_size, inner_size, bias=False)
        self.l2 = nn.Linear(hidden_size, inner_size, bias=False)
        self.l3 = nn.Linear(inner_size, hidden_size, bias=False)
        self.act = F.silu

    def forward(self, z):
        z1, z2 = self.l1(z), self.l2(z)
        return self.l3(self.act(z1) * z2)

class FeedForward(nn.Module):
    """ 표준 Transformer Feed-Forward Network """
    def __init__(self, embedding_dim, ff_hidden_dim, **kwargs):
        super().__init__()
        self.W1 = nn.Linear(embedding_dim, ff_hidden_dim)
        self.W2 = nn.Linear(ff_hidden_dim, embedding_dim)

    def forward(self, input1):
        return self.W2(F.relu(self.W1(input1)))

def reshape_by_heads(qkv: torch.Tensor, head_num: int) -> torch.Tensor:
    """ (batch, n, dim) -> (batch, head_num, n, key_dim) """
    batch_s, n = qkv.size(0), qkv.size(1)
    q_reshaped = qkv.reshape(batch_s, n, head_num, -1)
    q_transposed = q_reshaped.transpose(1, 2)
    return q_transposed

def multi_head_attention(q, k, v, ninf_mask=None, sparse=False):
    """ Multi-Head Attention 계산 함수 """
    batch_s, head_num, n, key_dim = q.shape
    input_s = k.size(2)
    score = torch.matmul(q, k.transpose(2, 3))
    score_scaled = score / (key_dim ** 0.5)

    if ninf_mask is not None:
        score_scaled = score_scaled + ninf_mask[:, None, :, :].expand(batch_s, head_num, n, input_s)

    # CaDA의 sparse 옵션은 POCAT에서는 단순화하거나 제거할 수 있습니다.
    # 여기서는 기본 Softmax만 사용하도록 수정합니다.
    weights = nn.Softmax(dim=3)(score_scaled)
    out = torch.matmul(weights, v)

    out_transposed = out.transpose(1, 2)
    out_concat = out_transposed.reshape(batch_s, n, head_num * key_dim)
    return out_concat

class EncoderLayer(nn.Module):
    """ Transformer 인코더의 단일 레이어 """
    def __init__(self, embedding_dim, head_num, qkv_dim, ffd='siglu', **model_params):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.head_num = head_num
        self.qkv_dim = qkv_dim
        
        # MHA 파라미터
        self.Wq = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wk = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.multi_head_combine = nn.Linear(head_num * qkv_dim, embedding_dim)
        
        self.normalization1 = Normalization(embedding_dim, **model_params)
        
        # FFD 파라미터
        if ffd == 'siglu':
            self.feed_forward = ParallelGatedMLP(hidden_size=embedding_dim, **model_params)
        else:
            self.feed_forward = FeedForward(embedding_dim=embedding_dim, **model_params)
            
        self.normalization2 = Normalization(embedding_dim, **model_params)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. Multi-Head Attention
        q = reshape_by_heads(self.Wq(x), self.head_num)
        k = reshape_by_heads(self.Wk(x), self.head_num)
        v = reshape_by_heads(self.Wv(x), self.head_num)
        
        mha_out = multi_head_attention(q, k, v)
        mha_out = self.multi_head_combine(mha_out)
        
        # 2. Add & Norm
        h = self.normalization1(x + mha_out)
        
        # 3. Feed Forward
        ff_out = self.feed_forward(h)
        
        # 4. Add & Norm
        out = self.normalization2(h + ff_out)
        
        return out
    


class PocatPromptNet(nn.Module):
    """ CaDA를 개조한 Power Tree 생성 모델 (POMO 적용) """
    def __init__(self, **model_params):
        super().__init__()
        embedding_dim = model_params['embedding_dim']
        self.prompt_net = PocatPromptNet(embedding_dim, prompt_feature_dim=2)
        self.encoder = PocatEncoder(embedding_dim=embedding_dim, **model_params)
        self.decoder = PocatDecoder(embedding_dim=embedding_dim, **model_params)
        self.context_gru = nn.GRUCell(embedding_dim * 2, embedding_dim)

        
    def forward(self, td: TensorDict, env: 'PocatEnv'):
        
        # 1. 인코딩
        prompt_embedding = self.prompt_net(td["prompt_features"])
        encoded_nodes = self.encoder(td["nodes"], prompt_embedding)

        # 2. POMO 시작점 선택 및 데이터 확장 (Batchify)
        num_starts, start_nodes_idx = env.select_start_nodes(td)
        
        # CaDA의 유틸리티 함수를 가져와서 사용
        
        td = batchify(td, num_starts)
        encoded_nodes = batchify(encoded_nodes, num_starts)
        
        # 3. 디코딩 준비
        batch_size = td.batch_size[0]
        num_nodes = td["nodes"].shape[1]
        context_embedding = encoded_nodes.mean(dim=1) # 초기 컨텍스트
        
        log_probs_list = []
        actions_list = []


class PocatEncoder(nn.Module):
    """ POCAT 노드들의 전기적 속성을 인코딩하는 Transformer 인코더 """
    def __init__(self, embedding_dim: int, num_layers: int = 6, **kwargs):
        super().__init__()
        # 노드의 11차원 피처를 embedding_dim으로 변환
        self.embedding_layer = nn.Linear(FEATURE_DIM, embedding_dim)
        self.layers = nn.ModuleList(
            [EncoderLayer(embedding_dim=embedding_dim, **kwargs) for _ in range(num_layers)]
        )

    def forward(self, td: TensorDict, env: 'PocatEnv'):
        
        # 1. 인코딩
        prompt_embedding = self.prompt_net(td["prompt_features"])
        encoded_nodes = self.encoder(td["nodes"], prompt_embedding)

        # 2. POMO 시작점 선택 및 데이터 확장 (Batchify)
        num_starts, start_nodes_idx = env.select_start_nodes(td)
        
        td = batchify(td, num_starts)
        encoded_nodes = batchify(encoded_nodes, num_starts)
        
        # 3. 디코딩 준비
        batch_size = td.batch_size[0]
        num_nodes = td["nodes"].shape[1]
        context_embedding = encoded_nodes.mean(dim=1) # 초기 컨텍스트
        
        log_probs_list = []
        actions_list = []

        # --- POMO를 위한 첫 번째 스텝 ---
        # 첫 액션: 주어진 시작 노드(자식)에 대한 부모 찾기
        
        # Decoder의 2단계 중 '부모 선택' 로직만 수행
        # Query: 컨텍스트 + 시작 노드(자식) 임베딩
        start_child_embedding = encoded_nodes[torch.arange(batch_size), start_nodes_idx]
        parent_query_input = torch.cat([context_embedding, start_child_embedding], dim=1)
        parent_query = self.decoder.parent_wq(parent_query_input).unsqueeze(1)
        parent_keys = self.decoder.parent_wk(encoded_nodes)
        
        parent_scores = torch.matmul(parent_query, parent_keys.transpose(1, 2)).squeeze(1)
        parent_scores /= (self.decoder.embedding_dim ** 0.5)

        # 마스크 적용
        mask = env.get_action_mask(td)
        parent_mask = mask[torch.arange(batch_size), start_nodes_idx]
        parent_scores[~parent_mask] = -float('inf')
        parent_log_probs = F.log_softmax(parent_scores, dim=-1)
        
        # 첫 부모 선택
        selected_parent_idx = parent_log_probs.argmax(dim=-1)
        
        # 첫 액션 정의 및 환경 업데이트
        first_action = torch.stack([start_nodes_idx, selected_parent_idx], dim=1)
        td.set("action", first_action)
        td = env.step(td)

        # 로그 확률 및 액션 저장
        first_log_prob = parent_log_probs[torch.arange(batch_size), selected_parent_idx]
        actions_list.append(first_action)
        log_probs_list.append(first_log_prob)
        
        # 컨텍스트 업데이트
        parent_emb = encoded_nodes[torch.arange(batch_size), selected_parent_idx]
        gru_input = torch.cat([start_child_embedding, parent_emb], dim=1)
        context_embedding = self.context_gru(gru_input, context_embedding)
        
        # --- 이후 디코딩 루프 ---
        for _ in range(1, num_nodes - 1): # 첫 스텝은 이미 진행했으므로 1부터 시작
            if td["done"].all():
                break

            mask = env.get_action_mask(td)
            log_prob, action = self.decoder(encoded_nodes, context_embedding, mask)
            
            td.set("action", action)
            actions_list.append(action)
            log_probs_list.append(log_prob)
            
            td = env.step(td)

            child_emb = encoded_nodes[torch.arange(batch_size), action[:, 0]]
            parent_emb = encoded_nodes[torch.arange(batch_size), action[:, 1]]
            gru_input = torch.cat([child_emb, parent_emb], dim=1)
            context_embedding = self.context_gru(gru_input, context_embedding)

        # 4. 결과 정리
        actions = torch.stack(actions_list, dim=1)
        log_likelihood = torch.stack(log_probs_list, dim=1).sum(dim=1)
        
        return {
            "reward": td["reward"],
            "log_likelihood": log_likelihood,
            "actions": actions
        }

class PocatDecoder(nn.Module):
    """ 2단계 Attention으로 (자식, 부모) 연결을 결정하는 디코더 """
    def __init__(self, embedding_dim: int, head_num: int = 8, **kwargs):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.head_num = head_num

        # 1단계: 자식 노드 선택을 위한 Attention
        self.child_wq = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.child_wk = nn.Linear(embedding_dim, embedding_dim, bias=False)

        # 2단계: 부모 노드 선택을 위한 Attention
        # Query: Global context + 선택된 자식 노드 정보
        self.parent_wq = nn.Linear(embedding_dim * 2, embedding_dim, bias=False)
        self.parent_wk = nn.Linear(embedding_dim, embedding_dim, bias=False)

    def forward(self, encoded_nodes: torch.Tensor, context_embedding: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            encoded_nodes: 인코더 출력 (batch, num_nodes, dim)
            context_embedding: 현재 Tree 상태를 요약한 컨텍스트 벡터 (batch, dim)
            mask: 유효한 (자식, 부모) 선택을 위한 마스크 (batch, num_nodes, num_nodes)

        Returns:
            log_probs: (자식, 부모) 쌍에 대한 로그 확률
            action: 선택된 (자식, 부모) 인덱스
        """
        # --- 1단계: 연결할 자식 노드 선택 ---
        child_query = self.child_wq(context_embedding).unsqueeze(1) # (B, 1, D)
        child_keys = self.child_wk(encoded_nodes) # (B, N, D)

        # Attention score 계산
        child_scores = torch.matmul(child_query, child_keys.transpose(1, 2)) # (B, 1, N)
        child_scores = child_scores.squeeze(1) / (self.embedding_dim ** 0.5)

        # 유효한 자식 노드만 선택하도록 마스킹
        child_mask = mask.any(dim=2) # 부모로 연결될 수 있는 노드가 하나라도 있으면 유효한 자식
        child_scores[~child_mask] = -float('inf')
        child_log_probs = F.log_softmax(child_scores, dim=-1)

        # 자식 노드 선택 (Greedy or Sampling)
        # 여기서는 간단하게 argmax (Greedy) 사용
        selected_child_idx = child_log_probs.argmax(dim=-1) # (B,)

        # --- 2단계: 자식을 연결할 부모 노드 선택 ---
        selected_child_embedding = encoded_nodes[torch.arange(encoded_nodes.shape[0]), selected_child_idx]

        # Query: 컨텍스트와 선택된 자식 정보를 결합
        parent_query_input = torch.cat([context_embedding, selected_child_embedding], dim=1)
        parent_query = self.parent_wq(parent_query_input).unsqueeze(1) # (B, 1, D)
        parent_keys = self.parent_wk(encoded_nodes) # (B, N, D)

        # Attention score 계산
        parent_scores = torch.matmul(parent_query, parent_keys.transpose(1, 2)) # (B, 1, N)
        parent_scores = parent_scores.squeeze(1) / (self.embedding_dim ** 0.5)

        # 유효한 부모 노드만 선택하도록 마스킹
        parent_mask = mask[torch.arange(mask.shape[0]), selected_child_idx]
        parent_scores[~parent_mask] = -float('inf')
        parent_log_probs = F.log_softmax(parent_scores, dim=-1)

        # 부모 노드 선택
        selected_parent_idx = parent_log_probs.argmax(dim=-1) # (B,)

        # 최종 선택된 액션과 로그 확률
        action = torch.stack([selected_child_idx, selected_parent_idx], dim=1)
        
        # 전체 행동에 대한 로그 확률 (단순 합으로 근사)
        child_prob = child_log_probs[torch.arange(child_log_probs.shape[0]), selected_child_idx]
        parent_prob = parent_log_probs[torch.arange(parent_log_probs.shape[0]), selected_parent_idx]
        log_probs = child_prob + parent_prob

        return log_probs, action


class PocatModel(nn.Module):
    """ CaDA를 개조한 Power Tree 생성 모델 (POMO 적용) """
    def __init__(self, **model_params):
        super().__init__()
        embedding_dim = model_params['embedding_dim']

        self.prompt_net = PocatPromptNet(embedding_dim, prompt_feature_dim=2)
        self.encoder = PocatEncoder(**model_params)
        self.decoder = PocatDecoder(**model_params)
        
        # 디코더의 컨텍스트 입력을 위한 GRU 셀
        self.context_gru = nn.GRUCell(embedding_dim * 2, embedding_dim)


    def forward(self, td: TensorDict, env: 'PocatEnv'):
        batch_size = td.batch_size[0]
        num_nodes = td["nodes"].shape[1]

        # 1. 인코딩
        prompt_embedding = self.prompt_net(td["prompt_features"])
        encoded_nodes = self.encoder(td["nodes"], prompt_embedding)
        
        # 2. 디코딩 준비
        # GRU의 hidden state 역할을 할 컨텍스트 벡터 초기화
        # 인코딩된 노드들의 평균값을 초기 컨텍스트로 사용
        context_embedding = encoded_nodes.mean(dim=1)
        
        log_probs_list = []
        actions_list = []

        # 3. 디코딩 루프 (Tree 건설)
        # Power Tree는 보통 모든 부하가 연결될 때까지 step을 진행함
        # 최대 step은 (전체 노드 수 - 1)
        for _ in range(num_nodes - 1):
            if td["done"].all():
                break

            mask = env.get_action_mask(td)
            
            # 디코더를 통해 다음 액션(연결) 선택
            log_prob, action = self.decoder(encoded_nodes, context_embedding, mask)
            
            # 액션과 로그 확률 저장
            td.set("action", action)
            actions_list.append(action)
            log_probs_list.append(log_prob)
            
            # 환경 step 진행
            td = env.step(td)

            # 컨텍스트 업데이트: 선택된 자식과 부모 노드 정보를 GRU에 입력
            child_emb = encoded_nodes[torch.arange(batch_size), action[:, 0]]
            parent_emb = encoded_nodes[torch.arange(batch_size), action[:, 1]]
            gru_input = torch.cat([child_emb, parent_emb], dim=1)
            context_embedding = self.context_gru(gru_input, context_embedding)

        # 4. 결과 정리
        actions = torch.stack(actions_list, dim=1)
        log_likelihood = torch.stack(log_probs_list, dim=1).sum(dim=1)
        
        return {
            "reward": td["reward"],
            "log_likelihood": log_likelihood,
            "actions": actions
        }