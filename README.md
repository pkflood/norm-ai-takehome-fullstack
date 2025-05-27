This repository contains a client and server codebase. 

## Server Repository:

This codebase contains a list of laws (`docs/laws.pdf`) taken from the fictional series “Game of Thrones” (randomly pulled from a wiki fandom site... unfortunately knowledge of the series does not provide an edge on this assignment). Your task is to implement a new service (described in take home exercise document) and provide access to that service via a FastAPI endpoint running in a docker container. Please replace this readme with the steps required to run your app.

## Client Repository 

In the `frontend` folder you'll find a light NextJS app with it's own README including instructions to run. Your task here is to build a minimal client experience that utilizes the service build in part 1.


## Steps to Run the Application:
#### Prerequisite: Have Docker Installed on your machine
### 1. Build Docker Image: 
Run the following command, where myapp is the name of the image you choose, to build the image. You must have the Dockerfile in the same root directory as the rest of the files in the project. 

    docker build -t myapp:latest .
### 2. Run Docker container from the image: 
Pass in your own openai api key, choose a name for the container (in this case norm-takehome), and use the same image name as step 1. 

    docker run -e OPENAI_API_KEY="" -d -p 8080:80 --name norm-takehome myapp:latest
### 3. Verify the image is running
The first command will show you whether or not the container is running or has exited, along with the container name and id. Run "docker logs" with the container id from the previous command to view the details of the app launch. 

    docker ps -a
    docker logs <container id>
### 3. In your browser navigate to this url:
    http://localhost:8080/docs 
### 4. View Website and Input Query
A swagger API UI should populate the screen. Once here, click on the "Try it Out" button on the top left to enable query inputs in the 
text box. Enter your query (ex."What happens to thieves?") and click "Execute." 
### 5. Get Response
The response body will populate with a JSON response showing the query, response, and citations. 