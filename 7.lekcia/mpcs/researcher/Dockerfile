# Use an official Python runtime as a parent image
FROM python:3.12-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install the `uv` package
RUN pip3 install --no-cache-dir uv
RUN uv sync

# Define the command to run your terminal-based application
CMD ["uv", "run", "main.py"]
