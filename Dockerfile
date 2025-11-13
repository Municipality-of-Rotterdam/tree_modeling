FROM mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu22.04
# Set env variables
ENV DEBIAN_FRONTEND=noninteractive
# Necessary OS installs
RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
    libxext6 \
    libsm6 \
    libgl1 \
    libgl1-mesa-glx \
    libopengl0 \
    pdal && \
    apt-get clean -y && \
    rm -rf /var/lib/apt/lists/*