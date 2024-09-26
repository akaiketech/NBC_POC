# Use the official Python image from the Docker Hub
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Install necessary system packages
RUN apt-get update && \
    apt-get install -y libgl1-mesa-glx libglib2.0-0 libsm6 libxrender1 libxext6 libgtk2.0-dev

# Copy the requirements file to the container
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code to the container
COPY . .

# Expose port 8078 to allow external connections
EXPOSE 8078

# Run the Streamlit application when the container launches
CMD ["streamlit", "run", "main.py", "--server.port", "8078"]
