# trainer.py
import torch
from tqdm import tqdm

# 💡 우리가 만든 클래스와 유틸리티 함수들을 임포트합니다.
from pocat_utils import TimeEstimator, clip_grad_norms, unbatchify
from model import PocatModel
from pocat_env import PocatEnv

def cal_model_size(model, log_func):
    param_count = sum(param.nelement() for param in model.parameters())
    buffer_count = sum(buffer.nelement() for buffer in model.buffers())
    log_func(f'Total number of parameters: {param_count}')
    log_func(f'Total number of buffer elements: {buffer_count}')

class PocatTrainer:
    def __init__(self, args, env: PocatEnv):
        self.args = args
        self.env = env
        
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        
        self.model = PocatModel(**args.model_params).to('cuda')
        cal_model_size(self.model, args.log)
        
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=args.optimizer_params['optimizer']['lr'],
            weight_decay=args.optimizer_params['optimizer'].get('weight_decay', 0),
        )
        
        if args.optimizer_params['scheduler']['name'] == 'MultiStepLR':
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
                self.optimizer,
                milestones=args.optimizer_params['scheduler']['milestones'],
                gamma=args.optimizer_params['scheduler']['gamma']
            )
        else:
            raise NotImplementedError
            
        self.start_epoch = 1
        # TODO: 체크포인트 로딩 로직 (필요시 CaDA 코드 참고하여 구현)

        self.time_estimator = TimeEstimator(logger=args.log)

    def run(self):
        args = self.args
        self.time_estimator.reset(self.start_epoch)
        
        if args.test_only: 
            print(">> POCAT 모델의 테스트 기능은 아직 구현되지 않았습니다.")
            return

        # 훈련 시작
        for epoch in range(self.start_epoch, args.trainer_params['epochs'] + 1):
            args.log('=================================================================')
            
            self.model.train()
            train_pbar = tqdm(range(args.trainer_params['train_step']), 
                              bar_format='{desc}|{elapsed}+{remaining}|{n_fmt}/{total_fmt}', 
                              leave=False, dynamic_ncols=True)
            train_label = f"Train|E{str(epoch).zfill(3)}/{args.trainer_params['epochs']}"
            
            for step in train_pbar:
                self.optimizer.zero_grad()
                
                td = self.env.reset(batch_size=args.batch_size)
                
                out = self.model(td, self.env)
                
                # POMO 결과를 원래 배치 형태로 변환 (unbatchify)
                num_starts = self.env.generator.num_loads
                reward = unbatchify(out["reward"], num_starts) # (B*L, 1) -> (B, L, 1)
                log_likelihood = unbatchify(out["log_likelihood"], num_starts) # (B*L) -> (B, L)

                # 가장 좋은 결과(가장 높은 보상)를 기준으로 학습
                best_reward, best_idx = reward.max(dim=1)
                
                # REINFORCE 알고리즘으로 Loss 계산
                advantage = best_reward - reward.mean(dim=1) # Baseline: 평균 보상
                
                # 가장 좋은 결과를 낸 경로의 log_likelihood를 사용
                best_log_likelihood = log_likelihood.gather(1, best_idx).squeeze(-1)
                
                loss = -(advantage * best_log_likelihood).mean()
                
                loss.backward()
                clip_grad_norms(self.optimizer.param_groups, 1.0)
                self.optimizer.step()
                
                # 평균 비용 (보상은 음수 비용)
                avg_cost = -best_reward.mean().item()
                train_pbar.set_description(f"🙏> {train_label}| Loss:{loss.item():.4f} Cost:{avg_cost:.4f}")

            self.scheduler.step()
            self.time_estimator.print_est_time(epoch, args.trainer_params['epochs'])
            
            # TODO: 모델 저장 로직 (필요시 CaDA 코드 참고하여 구현)

        args.log(" *** Training Done *** ")