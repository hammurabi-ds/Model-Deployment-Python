# Model-Deployment-RESTful
model deployment in flask restful

Steps for deployment:

1. Build the docker container by `docker build . -t APP_NAME`
2. Run the docker container by `docker run -d -p PORT --name CONTAINER_NAME APP_NAME`
