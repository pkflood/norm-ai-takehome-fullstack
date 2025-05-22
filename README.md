This repository contains a client and server codebase. 

## Server Repository:

This codebase contains a list of laws (`docs/laws.pdf`) taken from the fictional series “Game of Thrones” (randomly pulled from a wiki fandom site... unfortunately knowledge of the series does not provide an edge on this assignment). Your task is to implement a new service (described in take home exercise document) and provide access to that service via a FastAPI endpoint running in a docker container. Please replace this readme with the steps required to run your app.

## Client Repository 

In the `frontend` folder you'll find a light NextJS app with it's own README including instructions to run. Your task here is to build a minimal client experience that utilizes the service build in part 1.


## Steps to Run the Application: 
### 1.Run docker run (fill in later)
### 2. Launch the app
    uvicorn app.main:app --reload
### 3. In your browser navigate to this url:
    http://127.0.0.1:8000/docs#/default/query_query_get 
### 4. View Website and Input Query
A swagger API UI should populate the screen. Once here, click on the "Try it Out" button on the top left to enable query inputs in the 
text box. Enter your query (ex."What happens to theives?") and click "Execute." 
### 5. Get Response
The response body will populate with a JSON response showing the query, response, and citations. 