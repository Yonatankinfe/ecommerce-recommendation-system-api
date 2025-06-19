# Start with an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container at /app
# We'll create this requirements.txt in a moment, for now, this is a placeholder.
# Alternatively, copy Pipfile and Pipfile.lock if using pipenv
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
# --no-cache-dir: Disables the cache, which can reduce image size.
# -U: Upgrade all specified packages to the newest available version.
RUN pip install --no-cache-dir -U -r requirements.txt

# Copy the src directory and html directory into the container at /app
COPY ./src /app/src
COPY ./html /app/html

# Make port 8000 available to the world outside this container
EXPOSE 8000

# Define environment variable (optional, can be set in docker-compose.yml too)
# ENV NAME World

# Run main.py when the container launches
# Use uvicorn to run the FastAPI application.
# --host 0.0.0.0 makes the server accessible externally (from the host machine or other containers).
# --port 8000 matches the EXPOSE instruction.
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
