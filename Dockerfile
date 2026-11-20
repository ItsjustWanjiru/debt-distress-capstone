# API image: the trained model is copied in, so the container is ready to serve.
FROM python:3.13-slim
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY Data/__init__.py Data/data_pipeline.py Data/
COPY Model/__init__.py Model/components.py Model/train.py Model/model.joblib Model/metadata.json Model/
COPY API/ API/
EXPOSE 8000
CMD ["uvicorn", "API.main:app", "--host", "0.0.0.0", "--port", "8000"]
