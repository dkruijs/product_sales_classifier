FROM python:3.8-slim

COPY . /app
WORKDIR /app

ENV FLASK_APP=hello.py

RUN pip install --no-cache-dir -r requirements.txt

ENTRYPOINT ["flask", "run"]