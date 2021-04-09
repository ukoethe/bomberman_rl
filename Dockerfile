FROM continuumio/miniconda3
WORKDIR /home/bomberman
RUN conda install scipy numpy matplotlib numba
RUN conda install pytorch torchvision -c pytorch
RUN pip install scikit-learn tqdm tensorflow keras tensorboardX xgboost lightgbm
RUN pip install pathfinding
RUN conda install pandas
RUN pip install pydevd-pycharm~=211.6693.115
COPY . .
CMD /bin/bash
