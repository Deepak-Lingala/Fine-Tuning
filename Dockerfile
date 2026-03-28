FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV BASE_MODEL=Qwen/Qwen2.5-3B-Instruct
ENV ADAPTER_PATH=outputs/dpo-qwen2.5-3b-json

EXPOSE 8000

CMD ["uvicorn", "serve:app", "--host", "0.0.0.0", "--port", "8000"]
