FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY src ./src
# DATA=real trains on the ULB credit-card dataset (downloaded from OpenML at build time)
ARG DATA=synthetic
RUN python -m src.train --data ${DATA} && python -m src.registry promote
EXPOSE 8000
CMD ["sh", "-c", "uvicorn src.api:app --host 0.0.0.0 --port ${PORT:-8000}"]
