FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime

WORKDIR /workspace

COPY training /workspace/training

CMD ["python", "training/train.py"]