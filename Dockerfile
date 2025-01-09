# Menggunakan base image Python
FROM python:3.11-slim

# Set direktori kerja
WORKDIR /app

# Copy seluruh file ke dalam container
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port untuk aplikasi
EXPOSE 8080

# Command untuk menjalankan aplikasi menggunakan gunicorn
CMD ["gunicorn", "-b", "0.0.0.0:8080", "app:app"]
