FROM python:3.6.6-slim

# Mount the current directory to app in the container image
VOLUME ./:app/

# Copy local directory to /app in the container
COPY . /app/

# Change workdir
WORKDIR /app

# Install libraries
RUN apt-get update
RUN apt-get upgrade -y
RUN apt-get install -y python-numpy libicu-dev
RUN apt-get install -y gcc
RUN apt-get install -y --reinstall build-essential
RUN pip install -r requirements.txt
RUN python -m nltk.downloader popular
#RUN polyglot download LANG:en
RUN pip install gunicorn

# Expose the expose port and run the application when the container is spinned up
EXPOSE 9999
CMD ["gunicorn","-b 0.0.0.0:9999","wsgi:app"]

