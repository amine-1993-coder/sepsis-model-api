FROM python:3.10

# Set working directory
WORKDIR /app

# Copy all code to container
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port Connexion will use
EXPOSE 9090

# Run your app
CMD ["python", "app.py"]

