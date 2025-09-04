# utils/utils.py

import time
import sys
import os
import logging
import shutil

class TimeEstimator:
    """훈련 경과 및 남은 시간을 예측하여 출력하는 클래스"""
    def __init__(self):
        self.logger = logging.getLogger('TimeEstimator')
        self.start_time = time.time()
        self.count_zero = 0

    def reset(self, count=1):
        self.start_time = time.time()
        self.count_zero = count - 1

    def get_est_string(self, count, total):
        curr_time = time.time()
        elapsed_time = curr_time - self.start_time
        remain = total - count
        
        # 분모가 0이 되는 것을 방지
        if (count - self.count_zero) == 0:
            return "0m", "inf"
            
        remain_time = elapsed_time * remain / (count - self.count_zero)
        elapsed_time_h = elapsed_time / 3600.0
        remain_time_h = remain_time / 3600.0

        elapsed_time_str = f"{elapsed_time_h:.2f}h" if elapsed_time_h > 1.0 else f"{elapsed_time_h*60:.2f}m"
        remain_time_str = f"{remain_time_h:.2f}h" if remain_time_h > 1.0 else f"{remain_time_h*60:.2f}m"
        
        return elapsed_time_str, remain_time_str

def copy_all_src(dst_root: str):
    """
    훈련 시작 시, 'src' 폴더를 만들어 현재 실행에 사용된 모든 .py 소스 코드를 백업합니다.
    이를 통해 나중에 결과를 재현하거나 코드를 분석하기 용이해집니다.
    """
    try:
        execution_path = os.path.dirname(os.path.abspath(sys.argv[0]))
        dst_path = os.path.join(dst_root, 'src_backup')
        
        if not os.path.exists(dst_path):
            os.makedirs(dst_path)
            
        for root, _, files in os.walk(execution_path):
            for file in files:
                if file.endswith('.py'):
                    src_file_path = os.path.join(root, file)
                    # site-packages와 같은 외부 라이브러리는 복사하지 않음
                    if 'site-packages' in src_file_path or 'venv' in src_file_path:
                        continue
                    
                    relative_path = os.path.relpath(src_file_path, execution_path)
                    dst_file_path = os.path.join(dst_path, relative_path)
                    
                    os.makedirs(os.path.dirname(dst_file_path), exist_ok=True)
                    shutil.copy(src_file_path, dst_file_path)
        print(f"Source code backed up to: {dst_path}")
    except Exception as e:
        print(f"Could not back up source code: {e}")