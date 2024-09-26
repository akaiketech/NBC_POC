# Use the official Python image from the Docker Hub
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Install necessary dependencies for OpenCV or other libraries requiring system packages
RUN apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0 libsm6 libxrender1 libxext6

# Copy the requirements file to the container
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code to the container
COPY . .

# Expose port 8070 to allow external connections
EXPOSE 8078

# Run gunicorn when the container launches
# Start the FastAPI application
CMD ["streamlit", "run", "main.py", "--server.port", "8078"]


