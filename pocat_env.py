# pocat_env.py
import torch
from tensordict import TensorDict
from torchrl.envs import EnvBase
from typing import Optional, Tuple
from pocat_defs import (
    NODE_TYPE_BATTERY, NODE_TYPE_IC, NODE_TYPE_LOAD,
    FEATURE_INDEX
)

class PocatEnv(EnvBase):
    name = "pocat"

    def __init__(self, generator_params: dict = {}, device: str = "cpu", **kwargs):
        super().__init__(device=device)
        # PocatGenerator를 사용하도록 설정
        from pocat_generator import PocatGenerator
        self.generator = PocatGenerator(**generator_params)

    def _reset(self, td: Optional[TensorDict] = None, batch_size=None) -> TensorDict:
        if td is None:
            td = self.generator(batch_size=batch_size).to(self.device)

        # Power Tree 상태 초기화
        num_nodes = td["nodes"].shape[1]
        # 💡 수정된 부분: td["static_info"] 대신 self.generator.config에서 직접 정보를 가져옵니다.

        
        return TensorDict(
            {
                "nodes": td["nodes"],
                "prompt_features": td["prompt_features"],
                # "static_info": static_info_dict, # 이 라인 제거
                "connections": torch.zeros(td.batch_size[0], num_nodes - 1, 2, dtype=torch.long, device=self.device),
                "connected_nodes_mask": torch.zeros(td.batch_size[0], num_nodes, dtype=torch.bool, device=self.device),
                "ic_current_draw": torch.zeros(td.batch_size[0], num_nodes, device=self.device),
                "step_count": torch.zeros(td.batch_size[0], 1, dtype=torch.long, device=self.device),
                "done": torch.zeros(td.batch_size[0], 1, dtype=torch.bool, device=self.device),
            },
            batch_size=td.batch_size,
        )

    def _step(self, td: TensorDict) -> TensorDict:
        action = td["action"]  # action: (child_idx, parent_idx)
        child_idx, parent_idx = action[:, 0], action[:, 1]
        
        b_idx = torch.arange(td.batch_size[0])

        # 1. 상태 업데이트
        step = td["step_count"].squeeze(-1)
        # connections에 현재 액션(연결) 기록
        td["connections"][b_idx, step] = action

        # 자식과 부모 노드를 연결된 노드로 마스킹
        td["connected_nodes_mask"][b_idx, child_idx] = True
        # 배터리는 항상 연결 가능한 상태이므로 마스킹하지 않음 (is_battery 체크)
        parent_is_not_battery = td["nodes"][b_idx, parent_idx, FEATURE_INDEX["node_type"][0] + NODE_TYPE_BATTERY] != 1
        td["connected_nodes_mask"][b_idx, parent_idx[parent_is_not_battery]] = True

        # TODO: 전류량 계산 및 업데이트 (단순화를 위해 우선 생략)
        # 실제로는 child의 current_active를 parent의 ic_current_draw에 더하고,
        # 이 parent가 다른 IC의 자식이면 그 부모의 부모까지 거슬러 올라가며 전류량을 전파해야 함.

        # 2. 종료 조건 확인
        num_loads = self.generator.num_loads
        # 노드 피처에서 Load 타입인 노드들의 인덱스를 가져옴
        load_indices = torch.where(td["nodes"][0, :, FEATURE_INDEX["node_type"][0] + NODE_TYPE_LOAD] == 1)[0]

        all_loads_connected = td["connected_nodes_mask"][:, load_indices].all(dim=1)
        td["done"] = all_loads_connected.unsqueeze(-1)
        
        # 3. 보상 계산
        reward = self.get_reward(td)
        
        next_td = td.clone()
        next_td.update({
            "reward": reward,
            "step_count": td["step_count"] + 1,
        })
        return next_td


    def get_action_mask(self, td: TensorDict) -> torch.Tensor:
        """
        유효한 (자식, 부모) 연결만 선택할 수 있도록 마스크를 생성합니다.
        마스크 shape: (batch_size, num_nodes, num_nodes) -> mask[b, c, p] = 1 if (c,p) is valid
        """
        num_nodes = td["nodes"].shape[1]
        mask = torch.zeros(td.batch_size, num_nodes, num_nodes, dtype=torch.bool, device=self.device)
        
        b_idx = torch.arange(td.batch_size[0])

        # 1. 자식 노드 조건: 아직 연결되지 않은 IC 또는 Load
        node_types = td["nodes"][0, :, :FEATURE_INDEX["node_type"][1]].argmax(-1)
        is_ic_or_load = (node_types == NODE_TYPE_IC) | (node_types == NODE_TYPE_LOAD)
        unconnected_mask = ~td["connected_nodes_mask"]
        valid_child_mask = unconnected_mask & is_ic_or_load.unsqueeze(0)

        # 2. 부모 노드 조건: 배터리 또는 이미 연결된 IC
        is_battery_or_ic = (node_types == NODE_TYPE_BATTERY) | (node_types == NODE_TYPE_IC)
        # 배터리는 항상 연결 가능. IC는 connected_nodes_mask에 포함되어야 함
        can_be_parent_mask = (node_types == NODE_TYPE_BATTERY).unsqueeze(0) | td["connected_nodes_mask"]
        valid_parent_mask = can_be_parent_mask & is_battery_or_ic.unsqueeze(0)

        # 3. 전기적 제약 조건 (V_out vs V_in)
        # 부모의 Vout 범위가 자식의 Vin 요구사항을 만족해야 함
        parent_vout_min = td["nodes"][:, :, FEATURE_INDEX["vout_min"]].unsqueeze(1) # (B, 1, N)
        parent_vout_max = td["nodes"][:, :, FEATURE_INDEX["vout_max"]].unsqueeze(1) # (B, 1, N)
        child_vin_min = td["nodes"][:, :, FEATURE_INDEX["vin_min"]].unsqueeze(2)   # (B, N, 1)
        child_vin_max = td["nodes"][:, :, FEATURE_INDEX["vin_max"]].unsqueeze(2)   # (B, N, 1)
        
        voltage_ok = (parent_vout_min <= child_vin_max) & (parent_vout_max >= child_vin_min)
        
        # TODO: 전류 제약, 사이클 방지 등 추가 필요

        # 4. 최종 마스크 조합
        final_mask = (
            valid_child_mask.unsqueeze(2) &
            valid_parent_mask.unsqueeze(1) &
            voltage_ok
        )
        # 자기 자신에게 연결 방지
        final_mask[:, torch.arange(num_nodes), torch.arange(num_nodes)] = False
        
        return final_mask
        
    def get_reward(self, td: TensorDict) -> torch.Tensor:
        """ 보상을 계산합니다. """
        reward = torch.zeros(td.batch_size, device=self.device)
        done = td["done"].squeeze(-1)

        # 모든 부하가 연결된 상태(done)에서만 보상 계산
        if done.any():
            # 사용된 모든 노드 (= connected_nodes_mask가 True인 노드)
            used_nodes_mask = td["connected_nodes_mask"][done]
            
            # 그 중 IC인 노드들의 비용만 합산
            node_costs = td["nodes"][done, :, FEATURE_INDEX["cost"]]
            ic_mask = td["nodes"][done, :, FEATURE_INDEX["node_type"][0] + NODE_TYPE_IC] == 1
            
            # 실제로 사용된 IC들의 비용만 필터링
            used_ic_mask = used_nodes_mask & ic_mask
            total_cost = (node_costs * used_ic_mask).sum(dim=-1)
            
            reward[done] = -total_cost

        # TODO: 제약 위반 시 페널티 부여 로직 추가
        # 예: current_draw > i_limit 인 경우 reward -= 1000

        return reward.unsqueeze(-1)

    # 💡 추가된 _set_seed 메소드
    def _set_seed(self, seed: Optional[int]):
        """환경의 난수 생성기 시드를 설정합니다."""
        if seed is None:
            seed = torch.empty((), dtype=torch.int64).random_().item()
        rng = torch.manual_seed(seed)
        self.rng = rng

    def select_start_nodes(self, td: TensorDict) -> Tuple[int, torch.Tensor]:
        """
        POMO를 위해 탐색을 시작할 모든 'Load' 노드의 인덱스를 반환합니다.
        """
        # 첫 번째 배치 아이템을 기준으로 Load 노드들의 인덱스를 찾음
        node_types = td["nodes"][0, :, :FEATURE_INDEX["node_type"][1]].argmax(-1)
        load_indices = torch.where(node_types == NODE_TYPE_LOAD)[0]

        num_starts = len(load_indices)
        
        # (batch_size * num_starts) 형태로 확장
        start_nodes = load_indices.repeat_interleave(td.batch_size[0])
        
        return num_starts, start_nodes