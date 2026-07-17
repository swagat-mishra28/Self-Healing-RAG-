FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir sqlalchemy asyncpg psycopg2-binary uvicorn

# Copy app code
COPY . .

# Expose port
EXPOSE 8000

# Run app
CMD ["uvicorn", "backend.api_server:app", "--host", "0.0.0.0", "--port", "8000"]