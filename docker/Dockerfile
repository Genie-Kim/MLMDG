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

RUN pip install --user torch==1.2.0 torchvision==0.4.0
RUN pip install --user scipy scikit-learn scikit-image opencv-python tqdm Pillow yacs tensorflow nvidia-ml-py3 tensorboardX

RUN git config --global alias.lg "log --all --color --graph --pretty=format:'%Cred%h%Creset -%C(yellow)%d%Creset %s %Cgreen(%cr)%C(bold blue)<%an>%Creset' --abbrev-commit"
RUN git config --global alias.s "status -s"
RUN sudo apt-get update
RUN sudo apt-get install -y vim htop

RUN sudo apt install -y git curl python3-pip exuberant-ctags ack-grep
RUN sudo pip3 install pynvim flake8 pylint isort
RUN wget --quiet https://raw.github.com/fisadev/fisa-vim-config/v12.0.1/config.vim -O ~/.vimrc
