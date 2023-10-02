FROM ubuntu:20.04
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    pip \
    libgl1 \
    libglib2.0-0

RUN pip install -r requirements.txt