# Use an official Python runtime as a parent image
FROM python:3.8-slim

# Set the working directory to /app
WORKDIR /app

# Install system dependencies for machine learning and data processing libraries
RUN apt-get update && apt-get install -y \
    build-essential \
    libopenblas-dev \
    libomp-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements.txt first to leverage Docker caching
COPY requirements.txt /app/requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install PyTorch separately to ensure compatibility with the correct architecture
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Copy the rest of the application code
COPY . /app

# Expose port 5000 for Flask
EXPOSE 5000

# Command to run your Flask app
CMD ["python", "main.py"]