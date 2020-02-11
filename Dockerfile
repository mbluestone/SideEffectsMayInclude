# Dockerfile for SideEffectsMayInclude container

FROM ubuntu:18.04
MAINTAINER Max Bluestone <mbluestone93@gmail.com>

RUN apt-get update; \
    apt-get install -y software-properties-common; \
    apt-get install python3.6-dev curl -y; \
    apt-get install python3-pip -y;


# install python packages
RUN pip3 install --upgrade setuptools
RUN pip3 install \
requests \
matplotlib \
pandas \
numpy \
scipy \
dill \
beautifulsoup4 \
networkx \
pysmiles \
scikit-learn \
scikit-multilearn

# install torch packages
RUN pip3 install torch
RUN pip3 install torchvision
RUN pip3 install torchtext
RUN pip3 install --no-cache-dir torch-scatter
RUN pip3 install --no-cache-dir torch-sparse
RUN pip3 install --no-cache-dir torch-cluster
RUN pip3 install torch-geometric

COPY ./ /usr/local/SideEffectsMayInclude/

WORKDIR /usr/local/SideEffectsMayInclude/
