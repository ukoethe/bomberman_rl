FROM continuumio/miniconda3
WORKDIR /home/bomberman
RUN conda install scipy numpy matplotlib numba
RUN conda install pytorch torchvision -c pytorch
RUN pip install scikit-learn tqdm tensorflow keras tensorboardX xgboost lightgbm
COPY . .
CMD /bin/bash
