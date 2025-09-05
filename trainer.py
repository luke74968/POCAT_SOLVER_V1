# trainer.py
import torch
from tqdm import tqdm

from utils.common import TimeEstimator, clip_grad_norms, unbatchify
from model import PocatModel
from pocat_env import PocatEnv

def cal_model_size(model, log_func):
    param_count = sum(param.nelement() for param in model.parameters())
    buffer_count = sum(buffer.nelement() for buffer in model.buffers())
    log_func(f'Total number of parameters: {param_count}')
    log_func(f'Total number of buffer elements: {buffer_count}')

class PocatTrainer:
    # 💡 1. 생성자에서 device 인자를 받도록 수정
    def __init__(self, args, env: PocatEnv, device: str):
        self.args = args
        self.env = env
        self.device = device # 전달받은 device 저장
        
        # 💡 2. CUDA 강제 설정 라인 삭제
        # torch.set_default_tensor_type('torch.cuda.FloatTensor') 
        
        # 💡 3. 모델을 생성 후, 지정된 device로 이동
        self.model = PocatModel(**args.model_params).to(self.device)
        cal_model_size(self.model, args.log)
        
        # 💡 float()으로 감싸서 값을 숫자로 강제 변환합니다.
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=float(args.optimizer_params['optimizer']['lr']),
            weight_decay=float(args.optimizer_params['optimizer'].get('weight_decay', 0)),
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
        self.time_estimator = TimeEstimator(logger=args.log)

    def run(self):
        args = self.args
        self.time_estimator.reset(self.start_epoch)
        
        if args.test_only: 
            print(">> POCAT 모델의 테스트 기능은 아직 구현되지 않았습니다.")
            return

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
                
                num_starts = self.env.generator.num_loads
                reward = unbatchify(out["reward"], num_starts)
                log_likelihood = unbatchify(out["log_likelihood"], num_starts)
                
                best_reward, best_idx = reward.max(dim=1)
                advantage = best_reward - reward.mean(dim=1)
                best_log_likelihood = log_likelihood.gather(1, best_idx).squeeze(-1)
                
                loss = -(advantage * best_log_likelihood).mean()
                loss.backward()
                clip_grad_norms(self.optimizer.param_groups, 1.0)
                self.optimizer.step()
                
                avg_cost = -best_reward.mean().item()
                train_pbar.set_description(f"🙏> {train_label}| Loss:{loss.item():.4f} Cost:{avg_cost:.4f}")

            self.scheduler.step()
            self.time_estimator.print_est_time(epoch, args.trainer_params['epochs'])
            
        args.log(" *** Training Done *** ")