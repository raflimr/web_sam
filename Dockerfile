FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Install required system dependencies
RUN apt-get update && apt-get install -y \
    apt-transport-https \
    ca-certificates \
    gnupg \
    curl \
    libexpat1 \
    && rm -rf /var/lib/apt/lists/*

# Create necessary directories
RUN mkdir -p ./models/ \
    ./static/images/ \
    ./static/png/ \
    ./static/results/


# Download the model file
RUN curl -o ./models/sam_vit_b_01ec64.pth https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth

# Copy requirements and install dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code into the container
COPY . .

# Expose the Flask default port
EXPOSE 5000

# Command to run the application
CMD ["python", "app.py"]