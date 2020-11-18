FROM python:3.7-slim

WORKDIR /app

RUN apt-get update && \
     apt-get -y --no-install-recommends install \
     libgomp1

RUN pip install -U scikit-learn numpy xgboost

RUN apt update && apt -y install build-essential

RUN pip install pystan==2.18

RUN pip install fbprophet==0.5

COPY test_model.py ./test_model.py

ENTRYPOINT [ "python", "test_model.py" ]
