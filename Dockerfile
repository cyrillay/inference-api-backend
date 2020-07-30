FROM python:3.7-slim

MAINTAINER Cyril Lay "cyril@lays.pro"

COPY ./src /app/src
COPY ./requirements.txt /app/requirements.txt

WORKDIR /app

RUN pip3 install -r requirements.txt

##COPY . /app

ENTRYPOINT [ "python3", "-m", "src.run_keras_server" ]
