from tensorflow/tensorflow:latest-gpu
RUN apt update
RUN apt install git -y
RUN pip install numpy==1.21.5
RUN pip install pandas==1.3.5
RUN pip install scipy==1.7.3
RUN pip install python-decouple==3.6
RUN pip install ipykernel==6.9.1
RUN pip install matplotlib
RUN pip install scikit-learn
RUN pip install git+https://github.com/estebanvz/crypto_price
RUN pip install git+https://github.com/estebanvz/crypto_metrics