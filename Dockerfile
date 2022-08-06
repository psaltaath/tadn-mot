FROM pytorch/pytorch:1.12.0-cuda11.3-cudnn8-runtime
RUN apt-get update 
RUN apt-get install -y git
ADD . /tadn-mot/
RUN pip install -r /tadn-mot/requirements.txt
WORKDIR /tadn-mot