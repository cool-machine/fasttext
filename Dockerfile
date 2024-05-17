
# Use an official Python runtime as a parent image
FROM python:3.12-slim

# Set the working directory in the container
WORKDIR /app

# Copy only the requirements file first to leverage Docker cache
COPY requirements.txt /app/

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Install PyTorch CPU-only version
RUN pip install torch==2.3.0+cpu -f https://download.pytorch.org/whl/torch_stable.html

# Copy the rest of the application code
COPY . /app

# Download NLTK data
RUN python nltk_downloader.py

# Make port 8000 available to the world outside this container
EXPOSE 8000

# Run the FastAPI application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]


# # Use an official Python runtime as a parent image
# FROM python:3.12-slim

# # Set the working directory in the container
# WORKDIR /app

# # Copy only the requirements file first to leverage Docker cache
# COPY requirements.txt /app/

# # Install any needed packages specified in requirements.txt
# RUN pip install --no-cache-dir -r requirements.txt

# # Copy the rest of the application code
# COPY . /app

# # Download NLTK data
# RUN python nltk_downloader.py

# # Make port 8000 available to the world outside this container
# EXPOSE 8000

# # Run the FastAPI application
# CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]





# # Use an official Python runtime as a parent image
# FROM python:3.12-slim

# # Set the working directory in the container
# WORKDIR /app

# # Copy only necessary files and directories
# COPY requirements.txt /app/
# COPY nltk_downloader.py /app/
# COPY app /app/app

# # Optionally, copy specific model files
# COPY models /app/models/

# # Install any needed packages specified in requirements.txt
# # RUN pip install --no-cache-dir -r requirements.txt

# RUN pip install --no-cache-dir -r requirements.txt -v

# # Download NLTK data
# RUN python nltk_downloader.py--*

# # Make port 8000 available to the world outside this container
# EXPOSE 8000

# # Run app when the container launches
# CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

