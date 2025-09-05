#
Set-ExecutionPolicy Bypass -Scope Process

#
.\.venv\Scripts\Activate.ps1

# 예시 학습 시작 명령어
python run.py --config_file config.json --config_yaml config.yaml --batch_size 64


# 예시: "result/..." 부분과 "checkpoint-epoch-100.pth" 부분을 실제 경로와 파일명으로 바꿔주세요.
python run.py --test_only --load_path "result/2025-0905-171711/checkpoint-epoch-100.pth"
