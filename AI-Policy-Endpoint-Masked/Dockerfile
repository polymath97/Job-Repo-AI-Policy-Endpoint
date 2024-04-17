# Use the Python 3.11 base image
FROM python:3.11

# Setup chrome driver
RUN apt-get install -y wget unzip

# Set up the Chrome PPA
RUN wget -q -O - https://dl-ssl.google.com/linux/linux_signing_key.pub | apt-key add -
RUN echo "deb http://dl.google.com/linux/chrome/deb/ stable main" >> /etc/apt/sources.list.d/google.list

# Update the package list and install chrome
RUN apt-get update -y
RUN apt-get install -y google-chrome-stable

# Set up Chromedriver Environment variables
ENV CHROMEDRIVER_VERSION 2.19
ENV CHROMEDRIVER_DIR /chromedriver
RUN mkdir $CHROMEDRIVER_DIR

# Download and install Chromedriver
RUN wget -q --continue -P $CHROMEDRIVER_DIR "http://chromedriver.storage.googleapis.com/$CHROMEDRIVER_VERSION/chromedriver_linux64.zip"
RUN unzip $CHROMEDRIVER_DIR/chromedriver* -d $CHROMEDRIVER_DIR

# Put Chromedriver into the PATH
ENV PATH $CHROMEDRIVER_DIR:$PATH

# Setup dotnet library to convert the legacy office files extension
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    clang build-essential bash zlib1g-dev 

RUN mkdir -p /usr/share/dotnet && ln -s /usr/share/dotnet/dotnet /usr/bin/dotnet

RUN wget https://dot.net/v1/dotnet-install.sh \
    && chmod +x ./dotnet-install.sh \
    && ./dotnet-install.sh \
        --channel 8.0 \
        --install-dir /usr/share/dotnet

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file first to leverage Docker cache
COPY ./requirements.txt /app/requirements.txt

ENV DOTNET_SYSTEM_GLOBALIZATION_INVARIANT=true

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Copy the entire project directory into the container
COPY . /app

# Expose port 8080
EXPOSE 8080

# Command to start the FastAPI server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]