FROM python:3.12-slim

RUN pip install -U pip
RUN pip install pipenv 

WORKDIR /app

COPY [ "Pipfile", "Pipfile.lock", "./" ]

RUN pipenv install --system --deploy

COPY [ "predict.py", "linear_model.bin", "./" ]

EXPOSE 9696

ENV FLASK_APP=predict.py
ENV FLASK_RUN_PORT=9696
ENV FLASK_RUN_HOST=0.0.0.0
ENTRYPOINT [ "flask", "run" ]

