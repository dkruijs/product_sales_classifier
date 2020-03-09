FROM python:3.7-alpine

COPY . /app
WORKDIR /app

ENV FLASK_APP=hello.py

RUN pip install -r requirements.txt

ENTRYPOINT ["flask", "run"]