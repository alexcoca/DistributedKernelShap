FROM rayproject/autoscaler:ray-0.8.6
WORKDIR /distributed_explainers
COPY pyproject.toml .
COPY explainers ./explainers
RUN conda install python=3.7
RUN pip install .
