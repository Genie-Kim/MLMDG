FROM nvidia/cuda:10.0-cudnn7-runtime-ubuntu18.04

ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update && apt-get install -y \
	python3-opencv ca-certificates python3-dev git wget sudo  \
	cmake protobuf-compiler libprotobuf-dev && \
  rm -rf /var/lib/apt/lists/*
RUN ln -sv /usr/bin/python3 /usr/bin/python

# create a non-root user
ARG USER_ID=1000
RUN useradd -m --no-log-init --system  --uid ${USER_ID} jj -g sudo
RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers
USER jj
WORKDIR /home/jj

ENV PATH="/home/jj/.local/bin:${PATH}"
RUN wget https://bootstrap.pypa.io/get-pip.py && \
	python3 get-pip.py --user && \
	rm get-pip.py

ENV PATH ~/anaconda3/bin:$PATH
RUN wget --quiet https://repo.anaconda.com/archive/Anaconda3-2020.11-Linux-x86_64.sh -O ~/anaconda.sh && \
    /bin/bash ~/anaconda.sh -b -p ~/anaconda3 && \
    rm ~/anaconda.sh && \
    echo ". ~/anaconda3/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc
RUN source ~/.bashrc
RUN conda create -y -n mlmdg python=3.6
RUN conda activate mlmdg

RUN pip install --user torch==1.2.0 torchvision==0.4.0
RUN pip install --user scipy scikit-learn scikit-image opencv-python tqdm Pillow yacs tensorflow nvidia-ml-py3 tensorboardX
RUN conda install -y tsnecuda cuda100 -c cannylab

RUN git config --global alias.lg "log --all --color --graph --pretty=format:'%Cred%h%Creset -%C(yellow)%d%Creset %s %Cgreen(%cr)%C(bold blue)<%an>%Creset' --abbrev-commit"
RUN git config --global alias.s "status -s"
RUN sudo apt-get update
RUN sudo apt-get install -y vim htop
