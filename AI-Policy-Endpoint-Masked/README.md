# AI POLICY ENDPOINT

## RUNNING APPLICATION WITH VIRTUALENV
For the ease of use the package manager recommended is VirtualENV, you can use other package manager of your choice.

### Install virtualenv if you haven't installed it before
`$ python3 -m pip install virtualenv`

### Create and activate virtualenv for your project
In the project directory in the terminal you need to use python 3.11 for the environment, execute :

`$ python3.11 -m venv <you-env-name>`

Look at your folder, a new .env file will appear in the project directory. Modify the .env file by passing in the corresponding variable value like your openai api key, the path to embedding persist dir, model_name, and other variable

In the terminal, navigate to the newly created virtualenv folder. In your virtualenv folder execute this command

`$ source bin/activate`

If its successful you should see in your command prompt that your-env is displaying within parentheses

### Installing requirements
After you environment is activated, execute the following command to install the required library

`$ pip install -r requirements.txt`


### Running the endpoints
After all required libraries are installed and your virtual environment is active, execute this commmand on terminal

`$ uvicorn main:app --port 8080`

navigate to your browser with the following url

http://127.0.0.1:8080/docs

## RUNNING APPLICATION WITH DOCKER
if you have docker running in your computer, navigate to your project directory in the terminal and execute
`$ docker-compose up`

To access the container, open your browser and go to this url
http://0.0.0.0:8080/docs


## ENDPOINT USAGE GUIDE
The running application will have multiple endpoints, the parameter required for each endpoint is included within this repo in a file named "Endpoint Input Parameter" 
