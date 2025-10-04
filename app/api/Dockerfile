FROM python:3.9-slim

WORKDIR /app

COPY api/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY api/server.py .

EXPOSE 5000

CMD ["python", "server.py"]