FROM nvidia/cuda:12.5.0-runtime-ubuntu20.04

RUN apt-get update && apt-get install -y python3 python3-pip

RUN pip3 install flwr torch numpy matplotlib pandas torchvision scikit-learn seaborn

COPY client.py /app/client.py
COPY model.py /app/model.py
COPY train.py /app/train.py
COPY ../config.json /app/config.json

WORKDIR /app

CMD ["python3", "server.py"]