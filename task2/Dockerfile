# syntax=docker/dockerfile:1

FROM python:3.10-slim

WORKDIR /vc-similarity-app

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

COPY . .

RUN if [ -d chroma_data ]; then rm -rf chroma_data; fi

ARG OPENAI_API_KEY
ENV OPENAI_API_KEY=${OPENAI_API_KEY}

RUN python prepopulate_db.py

ENV FLASK_APP=VC-similarity-app.py

CMD [ "python", "-m" , "flask", "run", "--host=0.0.0.0"]
