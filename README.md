# Convolutional Neural Network for Language Detection

## About

This is a team school project for "Modern Tools of Development" class.  

## Installation

The project is tested on Windows x64 machines with Python 3.5.  

There is no explicit installation procedure, all you need is to fork this repo and install dependencies, which can be done by running  
```
pip install -r REQUIREMENTS.txt
```

## Usage

First step is to run flask server by executing  
```
python src/main.py
```
You should see something like that (note the address)  
```
* Running on http://127.0.0.1:5000/ (Press CTRL+C to quit)
```

Next, open `web/index.html` in browser.  
Ensure that `Backend-server` property matches the address you've seen before.  
Fill the form with sample text and press `[Detect language]` button to perform detection.  
If everything goes right, you will notice the plot and the scores updated.  
    
