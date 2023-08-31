# Use the mambaforge image as base image
FROM condaforge/mambaforge

# Set the working directory
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install the packages from environment.yml
RUN mamba env create -f environment.yml

# Install R, R's tidyverse, keras, and tensorflow in both R and Python
RUN /bin/bash -c "source activate nn_env && mamba install -c conda-forge r r-tidyverse r-keras tensorflow keras"

# Make port 80 available to the world outside this container
# EXPOSE 80
