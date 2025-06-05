FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .

# Add timeout and verbose output
RUN pip install --upgrade pip && \
    pip install -r requirements.txt --timeout 100 --no-cache-dir --verbose

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app.py"]
