from tensorflow/tensorflow:latest
MAINTAINER Aaron Zhao (yaz21@cam.ac.uk)
# Get dependencies
#RUN apt-get update && apt-get install -y \
#    php5-mcrypt \
#    python-pip
WORKDIR /root
COPY cache.py /root/cache.py
COPY cifar10.py /root/cifar10.py
COPY dataset.py /root/dataset.py
COPY download.py /root/download.py
COPY train.py /root/train.py

CMD ["python", "/root/train.py"]
