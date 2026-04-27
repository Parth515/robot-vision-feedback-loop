FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH="/app" 

COPY deployment/hf_space/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY deployment/hf_space/app.py .
COPY deployment/hf_space/README.md .

# Explicitly copy src and config from project root
COPY src/ ./src/
COPY config/ ./config/

EXPOSE 7860

CMD ["python", "app.py"]