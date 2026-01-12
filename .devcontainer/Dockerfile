FROM mcr.microsoft.com/devcontainers/python:1-3.12-bullseye
RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
    && apt-get -y install --no-install-recommends texlive-full \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Set up timezone using mounted host localtime
RUN ln -sf /usr/share/zoneinfo/hostlocaltime /etc/localtime