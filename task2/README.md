# Venture capital (VC) similarity app

See the task [here](https://docs.google.com/document/d/1--N-gnnHL_YOIbPVUr5imHCdVLwsCw4WFFNd7hL9lq8/edit) or read it duplicated below.

## Task description

You are working in a startup whose product is growing quite fast recently.
Based on the success, the CEO decided to start fundraising from Venture Capital firms.
But to do that, he asked you to first create a database of VCs that he can then reach out to. 
Your task is to build a Generative AI assistant for the CEO, which will be able to assess VC similarity and perform information extraction.

Given VC website URL as an input by the user, your Generative AI assistant should be able to:
* Scrape Home page of the website and store the content in vectorDB of your choice,
* Extract the following information and show it to the user as a JSON object: VC name, contacts, industries that they invest in, investment rounds that they participate/lead.
* Compare VC website content to the other VC website contents in the database (some examples provided below) and print the 3 most similar VCs.

Technical Requirements:
* Build the Generative AI assistant (e.g. using OpenAI APIs),
* Build an API (using FastAPI/Flask) that can be used as an interface for the AI Assistant,
* Either host the vectorDB, and the assistant app on cloud or containerize using Docker,
* Submit GitHub URL with your codebase. If the app is deployed in the cloud, submit the app URL as well.

Example VC websites that you can populate the database initially to compute the similarity:
| www.accel.com | www.a16z.com | www.greylock.com | www.benchmark.com |
| ------------- | ------------ | ---------------- | ----------------- |
| www.sequoiacap.com | www.indexventures.com | www.kpcb.com | www.lsvp.com |
| www.matrixpartners.com | www.500.co | www.sparkcapital.com | www.insightpartners.com |

## Solution

We are using the following frameworks:
* Flask - for wraping the backend in a web interface;
* Chroma DB - as vector database;
* OpenAI API - for the AI part; particularly `gpt-3.5-turbo` model is used for information extraction and `text-embedding-ada-002` for embedding texts.

Docker is used to contain the app.

First, for the database population the script `prepopulate_db.py` is run which fetches and parses the data of several VC websites and stores them in the database.
The database directory is `chroma_data`.
Then only the actual Flask application can operate. For a given VC website URL the app fetches and parses its data, then queries the database for the three closest matches and displays them.

## Run the application

Clone this repository. Navigate to the root of this project, i.e. `task2`. Make a virtual environment:
```
python3 -m venv venv
```
To activate the virtual environment run
* in Linux and MacOS
  ```sh
  source venv/bin/activate
  ```
* in Windows PowerShell
  ```bat
  venv\Scripts\activate
  ```

Note that you will need an OpenAI API key to run the app.
If you don't have one then you can create it [here](https://platform.openai.com/api-keys).
Once you have the key set the `OPENAI_API_KEY` environment variable.

After this you can run the app either in docker or locally.

### Using docker

Build the docker image by specifying your OpenAI API key:
* in Linux and MacOS
  ```sh
  docker build --build-arg OPENAI_API_KEY=$OPENAI_API_KEY -t vc-similarity-app .
  ```
* in Windows PowerShell
  ```bat
  docker build --build-arg OPENAI_API_KEY=$Env:OPENAI_API_KEY -t vc-similarity-app .
  ```

Run the app:
```
docker run -d -p 5000:5000 vc-similarity-app
```

To stop the app run `docker ps` to find out the container id and then run `docker stop <container_id>`.


### Running locally

Install the requirements:
```
pip install -r requirements.txt
```

Remove the directory `chroma_data` if it exists:
```
rm -r chroma_data
```

Populate the database:
```
python prepopulate.py
```

Define the flask app:
* in Linux and MasOS
  ```sh
  export FLASK_APP=VC-similarity-app.py
  ```
* in Windows PowerShell
  ```bat
  $Env:FLASK_APP = "VC-similarity-app.py"

Run the app:
```
flask run
```
