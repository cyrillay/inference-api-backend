FROM python:3.7-slim

MAINTAINER Cyril Lay "cyril@lays.pro"

##COPY ./requirements.txt /app/requirements.txt
COPY . /app
WORKDIR /app

RUN pip3 install -r requirements.txt

##COPY . /app

ENTRYPOINT [ "python3", "run_keras_server.py" ]
