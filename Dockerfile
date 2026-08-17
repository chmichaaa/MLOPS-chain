FROM python:3.10-slim-buster

RUN pip install --upgrade pip

WORKDIR /app

COPY . /app

ENV PYTHONPATH "${PYTHONPATH}:/app/prediction_model"

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8005

ENTRYPOINT ["python"]

CMD ["main.py"]
