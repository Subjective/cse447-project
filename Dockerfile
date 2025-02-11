FROM pytorch/pytorch:latest
RUN mkdir /job
WORKDIR /job
VOLUME ["/job/data", "/job/src", "/job/work", "/job/output"]

# You should install any dependencies you need here.
RUN pip install tqdm fasttext-wheel scikit-learn numpy urllib3
