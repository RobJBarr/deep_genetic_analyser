# How to run locally
Create an environment
>python3 -m venv venv

Activate the environment (Windows)
>venv\Scripts\activate

Activate the environment (Linux/OSX)
>. venv/bin/activate

Install Flask within activated environment
>pip3 install Flask

Install Requirements
>pip3 install -r requirements.txt

If on CMD (Windows)
> set FLASK_APP=app.py
> flask run

If on Bash (Linux/OSX)
>export FLASK_APP=app
>flask run