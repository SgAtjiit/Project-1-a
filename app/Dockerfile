# Use a lean base image for smaller size and faster builds
FROM --platform=linux/amd64 python:3.9-slim-buster

# Set the working directory in the container
WORKDIR /app

# Copy requirements.txt and install dependencies
# This step is cached if requirements.txt doesn't change
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install NLTK data required by the HeadingClassifier
RUN python -m nltk.downloader punkt stopwords

# Copy the application code
COPY app/ app/

# Command to run your application
# This will be overridden by the docker run command provided in the challenge
CMD ["python", "app/main.py"]