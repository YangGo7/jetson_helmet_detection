# jetson_helmet_detection
about jetson ai _ test
This project uses YOLOv5 to detect helmet usage at construction sites. It leverages NVIDIA Jetson devices for real-time deployment.
![Helmet Detection Example]![train_batch0](https://github.com/user-attachments/assets/e928270f-0448-44bb-ba15-0189294db7b9)
## Features
- Helmet detection with YOLOv5
- Optimized for NVIDIA Jetson devices
- Real-time inference
## Prerequisites
- Python 3.8 이상
- NVIDIA GPU (CUDA 지원)
- NVIDIA Jetson 장치 (Nano, Xavier NX 등) (선택 사항)
- Docker (선택 사항)
# 가상 환경 생성 (선택 사항)
python -m venv venv
# 가상 환경 활성화
# Windows
venv\Scripts\activate
# Linux/MacOS
source venv/bin/activate
# 종속성 설치
pip install -r requirements.txt
# PyTorch 설치 (CUDA 12.6 버전 지원)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
# OpenCV 설치
pip install opencv-python
# YOLOv5 레포지토리 클론
git clone https://github.com/ultralytics/yolov5.git
cd yolov5
# 종속성 설치
pip install -U -r requirements.txt
# 모델 학습 (예: 640x640 이미지 크기, 배치 크기 16, 10 에폭)
python train.py --img 640 --batch 16 --epochs 10 --data ./data.yaml --weights yolov5s.pt
# 학습된 모델로 이미지 추론
python detect.py --weights runs/train/exp/weights/best.pt --img 640 --source path/to/images
# 데이터셋 주소
https://universe.roboflow.com/insanudin-skhkb/object-detection-f1l7m \
## 테스트 및 평가
학습 후, 다음 명령어로 모델을 평가할 수 있습니다:
```bash
python test.py --weights runs/train/exp/weights/best.pt --data data.yaml

## 라이선스
이 프로젝트는 MIT 라이선스 하에 제공됩니다 - 자세한 내용은 [LICENSE](LICENSE) 파일을 참조하세요.
