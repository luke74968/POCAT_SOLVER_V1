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
    # CPU/GPU 자동 감지 로직
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.log(f"Using device: {device}")
    
    env = PocatEnv(
        generator_params={"config_file_path": args.config_file},
        device=device,
        instance_repeats=args.instance_repeats,
    )
    trainer = PocatTrainer(args, env, device)

    # 💡 --test_only 플래그가 있으면 훈련 대신 test() 함수를 실행
    if args.test_only:
        trainer.test()
    else:
        trainer.run()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 훈련 관련 인자
    parser.add_argument("--batch_size", type=int, default=64, help="Training batch_size")
    parser.add_argument(
        "--instance_repeats",
        type=int,
        default=1,
        help="Number of times to clone each instance",
    )    
    parser.add_argument("--config_file", type=str, default="config.json", help="Path to POCAT config file")
    parser.add_argument("--config_yaml", type=str, default="config.yaml", help="Path to model/training config YAML")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed")
    
    # 💡 추론을 위한 인자 추가
    parser.add_argument('--test_only', action='store_true', help="Only run test/inference")
    parser.add_argument('--load_path', type=str, default=None, help="Path to a saved model checkpoint (.pth)")

    args = parser.parse_args()
    
    args.start_time = time.strftime("%Y-%m%d-%H%M%S", time.localtime())
    args.result_dir = os.path.join('result', args.start_time)
    os.makedirs(args.result_dir, exist_ok=True)
    
    logger = setup_logger(args.result_dir)
    args.log = logger.info
    
    with open(args.config_yaml, "r", encoding="utf-8") as f:
        cfg_yaml = yaml.safe_load(f)
    for key, value in cfg_yaml.items():
        if not hasattr(args, key):
            setattr(args, key, value)

    args.ddp = False
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # 💡 수정된 부분: 로깅 전 non-JSON-serializable 객체 제거
    args_dict_for_log = vars(args).copy()
    del args_dict_for_log['log']
    args.log(json.dumps(args_dict_for_log, indent=4))
    
    main(args)