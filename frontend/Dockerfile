# Use a Python base image
FROM python:3.10-slim

# Set the working directory
WORKDIR /app

# Install dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy the Gradio application code
COPY . .

# Expose the Gradio port
EXPOSE 7860

# Command to run the application
CMD ["python", "app.py"]
