# Use an official Python runtime as a parent image
FROM python:3.9

# Set the working directory to /app
WORKDIR /app

# Copy only the requirements file to leverage Docker layer caching
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . .

# Expose the required ports
EXPOSE 8080
EXPOSE 9090

# Run the application when the container launches
CMD ["sh", "run.sh"]
