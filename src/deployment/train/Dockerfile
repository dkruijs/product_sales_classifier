FROM python:3.7-alpine

COPY . /app
WORKDIR /app

#RUN apk update && \
#    apk upgrade && \
#    apk add swig python-dev

RUN apk --update add --no-cache g++

RUN pip install -r requirements.txt