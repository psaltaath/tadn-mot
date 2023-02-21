FROM pytorch/pytorch:1.12.0-cuda11.3-cudnn8-runtime
RUN apt-get update 
RUN apt-get install -y build-essential libssl-dev libffi-dev
RUN apt-get install -y libgl1 libglib2.0-0
RUN apt-get install -y git
ADD . /tadn-mot/
RUN pip install -r /tadn-mot/requirements.txt
WORKDIR /tadn-mot