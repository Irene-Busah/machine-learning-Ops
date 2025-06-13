FROM agrigorev/zoomcamp-model:mlops-2024-3.10.13-slim

WORKDIR /app
COPY starter.py .

RUN pip install pandas pyarrow scikit-learn requests
ENTRYPOINT ["python", "starter.py"]
