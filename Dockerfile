# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Install Java, which is required by PySpark
RUN apt-get update && \
    apt-get install -y openjdk-11-jre-headless && \
    apt-get clean

# Copy the requirements file into the container
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application's code into the container
COPY . .

# Make port 8000 available to the world outside this container
EXPOSE 8000

# Define the command to run your app
# It will run 'uvicorn api:app' when the container launches
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
