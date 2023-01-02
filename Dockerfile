FROM continuumio/miniconda3
WORKDIR /home/bomberman
RUN apt-get update
RUN apt-get -y install gcc g++
RUN conda install scipy numpy matplotlib numba
RUN conda install pytorch torchvision -c pytorch
RUN pip install scikit-learn tqdm tensorflow keras tensorboardX xgboost lightgbm
RUN pip install pathfinding
RUN conda install pandas
RUN pip install networkx dill pyastar2d easydict sympy pygame
RUN     apt-get -y install x11vnc xvfb firefox-esr
RUN     mkdir ~/.vnc
RUN     x11vnc -storepasswd ibomber ~/.vnc/passwd
COPY . .
#ENTRYPOINT ["x11vnc", "-forever", "-usepw", "-create","&","python","main.py","play"]
CMD /bin/bash
