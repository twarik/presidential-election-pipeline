FROM python:3.7-slim

WORKDIR /app

RUN pip install -U scikit-learn numpy pandas imbalanced-learn

COPY preprocess_data.py ./preprocess_data.py

ENTRYPOINT [ "python", "preprocess_data.py" ]
