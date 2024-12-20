#!/bin/bash
source venv/bin/activate
export FLASK_APP=imymaxgpt.py
/home/max/imymaxgpt/venv/bin/flask run -h 0.0.0.0 -p 8083
