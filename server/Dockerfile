# Base Docker image
FROM python:3.9-slim

RUN apt-get update -y
RUN apt-get install ffmpeg libsm6 libxext6 -y
RUN pip install --upgrade pip

# Install Dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy all scripts
COPY . .

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]