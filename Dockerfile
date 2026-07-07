FROM python:3.12-slim

WORKDIR /app

RUN apt-get update
RUN apt-get install -y libgl1 libglib2.0-0

COPY requirements.txt .
RUN python -m pip install --upgrade pip
RUN python -m pip install -r requirements.txt

COPY src ./src
COPY backend ./backend
COPY configs ./configs
COPY frontend ./frontend

RUN mkdir -p /app/models

ENV MAX_UPLOAD_MB=12
EXPOSE 8000

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
