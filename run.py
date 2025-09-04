# run.py
import os
import sys
import time
import yaml
import json
import random
import torch
import logging
import argparse

from trainer import PocatTrainer
from pocat_env import PocatEnv

def setup_logger(result_dir):
    log_file = os.path.join(result_dir, 'log.txt')
    logging.basicConfig(filename=log_file, format='%(asctime)-15s %(message)s', level=logging.INFO)
    logger = logging.getLogger()
    console = logging.StreamHandler(sys.stdout)
    logger.addHandler(console)
    return logger

def main(args):
    # 환경과 트레이너 초기화
    env = PocatEnv(generator_params={'config_file_path': args.config_file}, device='cuda')
    trainer = PocatTrainer(args, env)
    
    # 훈련 시작
    trainer.run()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 훈련 관련 인자
    parser.add_argument("--batch_size", type=int, default=64, help="Training batch_size per gpu")
    parser.add_argument("--config_file", type=str, default="config.json", help="Path to POCAT config file")
    parser.add_argument("--config_yaml", type=str, default="config.yaml", help="Path to model/training config YAML")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed")
    parser.add_argument('--test_only', action='store_true', help="Only run test")
    
    args = parser.parse_args()
    
    # 결과 저장을 위한 디렉토리 생성
    args.start_time = time.strftime("%Y-%m%d-%H%M%S", time.localtime())
    args.result_dir = os.path.join('result', args.start_time)
    os.makedirs(args.result_dir, exist_ok=True)
    
    # 로거 설정
    logger = setup_logger(args.result_dir)
    args.log = logger.info
    
    # YAML 설정 파일 로드
    with open(args.config_yaml, "r") as f:
        cfg_yaml = yaml.safe_load(f)
    for key, value in cfg_yaml.items():
        if not hasattr(args, key):
            setattr(args, key, value)

    # DDP (Distributed Data Parallel) 설정은 우선 제외 (필요시 CaDA 코드 참고)
    args.ddp = False
    torch.cuda.set_device(0)
    
    # 랜덤 시드 고정
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    args.log(json.dumps(vars(args), indent=4))
    
    main(args)