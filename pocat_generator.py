# pocat_generator.py
import json
import torch
from tensordict import TensorDict
from typing import List
from pocat_defs import (
    PocatConfig, NODE_TYPE_BATTERY, NODE_TYPE_IC, NODE_TYPE_LOAD,
    FEATURE_DIM, FEATURE_INDEX
)

class PocatGenerator:
    """
    pocat_solver의 config.json 파일을 읽어,
    Transformer 모델 학습에 필요한 TensorDict 형태의 데이터를 생성합니다.
    """
    def __init__(self, config_file_path: str):
        with open(config_file_path, 'r') as f:
            config_data = json.load(f)
        self.config = PocatConfig(**config_data)
        self.num_nodes = len(self.config.node_names)
        self.num_loads = len(self.config.loads)

    def _create_feature_tensor(self) -> torch.Tensor:
        """ 각 노드(배터리, IC, 부하)의 전기적 속성을 텐서로 변환합니다. """
        features = torch.zeros(self.num_nodes, FEATURE_DIM)

        # 1. Battery Features
        battery_conf = self.config.battery
        features[0, FEATURE_INDEX["node_type"][0] + NODE_TYPE_BATTERY] = 1.0
        features[0, FEATURE_INDEX["vout_min"]] = battery_conf['voltage_min']
        features[0, FEATURE_INDEX["vout_max"]] = battery_conf['voltage_max']

        # 2. IC Features
        start_idx = 1
        for i, ic_conf in enumerate(self.config.available_ics):
            idx = start_idx + i
            features[idx, FEATURE_INDEX["node_type"][0] + NODE_TYPE_IC] = 1.0
            features[idx, FEATURE_INDEX["cost"]] = ic_conf.get('cost', 0.0)
            features[idx, FEATURE_INDEX["vin_min"]] = ic_conf.get('vin_min', 0.0)
            features[idx, FEATURE_INDEX["vin_max"]] = ic_conf.get('vin_max', 100.0)
            features[idx, FEATURE_INDEX["vout_min"]] = ic_conf.get('vout_min', 0.0)
            features[idx, FEATURE_INDEX["vout_max"]] = ic_conf.get('vout_max', 100.0)
            features[idx, FEATURE_INDEX["i_limit"]] = ic_conf.get('i_limit', 0.0)

        # 3. Load Features
        start_idx += len(self.config.available_ics)
        for i, load_conf in enumerate(self.config.loads):
            idx = start_idx + i
            features[idx, FEATURE_INDEX["node_type"][0] + NODE_TYPE_LOAD] = 1.0
            # 부하는 비용이 없으므로 cost는 0
            features[idx, FEATURE_INDEX["vin_min"]] = load_conf['voltage_req_min']
            features[idx, FEATURE_INDEX["vin_max"]] = load_conf['voltage_req_max']
            # 부하는 출력이 없으므로 vout은 0
            features[idx, FEATURE_INDEX["current_active"]] = load_conf['current_active']
            features[idx, FEATURE_INDEX["current_sleep"]] = load_conf['current_sleep']

        return features

    def __call__(self, batch_size: int) -> TensorDict:
        node_features = self._create_feature_tensor()
        constraints = self.config.constraints
        prompt_features = torch.tensor([
            constraints.get('ambient_temperature', 25.0),
            constraints.get('max_sleep_current', 0.0)
        ])

        node_features = node_features.unsqueeze(0).expand(batch_size, -1, -1)
        prompt_features = prompt_features.unsqueeze(0).expand(batch_size, -1)

        # 💡 수정된 부분: static_info를 TensorDict에서 제거
        return TensorDict(
            {
                "nodes": node_features,
                "prompt_features": prompt_features,
            },
            batch_size=[batch_size],
        )