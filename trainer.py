# trainer.py

import os
import random
import wandb
import torch
from tqdm import tqdm
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

# 💡 1. 우리가 만든 클래스와 필요한 유틸리티 함수들만 임포트합니다.
from utils.utils import TimeEstimator
from utils.functions import clip_grad_norms
from model import PocatModel
from pocat_env import PocatEnv

def cal_model_size(model, args):
    param_count = sum(param.nelement() for param in model.parameters())
    buffer_count = sum(buffer.nelement() for buffer in model.buffers())
    args.log(f'Total number of parameters: {param_count}')
    args.log(f'Total number of buffer elements: {buffer_count}')

class PocatTrainer:
    def __init__(self, args, env: PocatEnv):
        self.args = args
        self.env = env
        
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        
        # PocatModel을 생성합니다.
        self.model = PocatModel(**args.model_params)
        cal_model_size(self.model, args)
        
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr = args.optimizer_params['optimizer']['lr'],
            weight_decay=args.optimizer_params['optimizer'].get('weight_decay', 0),
        )
        if args.optimizer_params['scheduler']['name'] == 'MultiStepLR':
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
                self.optimizer,
                milestones = args.optimizer_params['scheduler']['milestones'],
                gamma = args.optimizer_params['scheduler']['gamma']
            )
        else:
            raise NotImplementedError
            
        self.start_epoch = 1
        model_load = args.trainer_params['model_load']
        if model_load['enable']:
            # 체크포인트 로딩 로직 (필요시 구현)
            pass

        if args.ddp:
            torch.distributed.barrier()
            self.model = DistributedDataParallel(self.model)
            for param in self.model.parameters():
                dist.broadcast(param.data, src=0)
            args.log(f'DDP enabled on device:{torch.cuda.current_device()}')

        self.time_estimator = TimeEstimator()

    def run(self):
        args = self.args
        self.time_estimator.reset(self.start_epoch)
        
        if args.test_only: 
            print("POCAT 모델의 테스트 기능은 아직 구현되지 않았습니다.")
            return

        # 훈련 시작
        for epoch in range(self.start_epoch, args.trainer_params['epochs'] + 1):
            args.log('=================================================================')
            
            self.model.train()
            train_pbar = tqdm(range(args.trainer_params['train_step']), bar_format='{desc}|{elapsed}+{remaining}|{n_fmt}/{total_fmt}', leave=False)
            train_label = f"Train|Epoch{str(epoch).zfill(3)}/{str(args.trainer_params['epochs']).zfill(3)}"
            
            for step in train_pbar:
                self.optimizer.zero_grad()
                
                # 환경에서 새로운 문제 생성
                td = self.env.reset(batch_size=args.batch_size)
                
                if args.ddp: torch.distributed.barrier()
                
                # 모델을 통해 Power Tree 생성
                out = self.model(td, self.env)
                
                reward = out["reward"]
                log_likelihood = out["log_likelihood"]
                
                # REINFORCE 알고리즘으로 Loss 계산
                advantage = reward - reward.mean() # Baseline: 평균 보상
                loss = -(advantage * log_likelihood).mean()
                
                # Score (평균 보상 = -평균 비용)
                score_mean = reward.mean().item()
                
                if args.ddp: torch.distributed.barrier()
                
                loss.backward()
                grad_norms, _ = clip_grad_norms(self.optimizer.param_groups, 1.0)
                self.optimizer.step()
                
                train_pbar.set_description(f"🙏> {train_label}| Loss:{loss.item():.4f} Cost:{-score_mean:.4f}")

            self.scheduler.step()
            
            # 로그 및 모델 저장 로직 (생략)

        args.log(" *** Training Done *** ")

    @torch.no_grad()
    def test(self, epoch):
        # POCAT용 테스트 로직은 VRP와 다르므로 새로 구현해야 합니다.
        self.args.log(f"Epoch {epoch}: POCAT test function not implemented yet.")
        pass