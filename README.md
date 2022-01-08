# How to run app locally
1. Change to directory
> cd deep_genetic_analyser 

2. Create an environment
>python3 -m venv venv

3a. Activate the environment (Windows)
>venv\Scripts\activate

3b. Activate the environment (Linux/OSX)
>. venv/bin/activate

4. Install Flask within activated environment
>pip3 install flask

5. Install Requirements
>pip3 install -r requirements.txt

6a. If on CMD (Windows)
> set FLASK_APP=app.py

> flask run

6b. If on Bash (Linux/OSX)
>export FLASK_APP=app

>flask run

7. Switch to a new terminal

8. Run on react
> npm install

> npm start