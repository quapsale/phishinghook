# Use official lightweight Python image
FROM python:3.10-slim

# Set working directory
WORKDIR /execution

# Copy all files into the container
COPY . .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Set the entrypoint to your main Python script
ENTRYPOINT ["python", "phishinghook.py"]
