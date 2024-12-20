from flask import Flask
from config import Config
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from flask_login import LoginManager
from flask_bootstrap import Bootstrap
from openai import OpenAI
import socket
socket.setdefaulttimeout(600) # seconds
from dotenv import load_dotenv
load_dotenv()

app = Flask(__name__)
app.config.from_object(Config)
db = SQLAlchemy(app)
migrate = Migrate(app, db)
#migrate = Migrate(app, db, render_as_batch=True)
login = LoginManager(app)
login.login_view = 'login'
login.login_message = "Пожалуйста, войдите, чтобы открыть эту страницу."

brand = app.config['BRAND']
brand_gpt = app.config['BRAND_GPT']

gl_api_key = app.config['OPENAI_API_KEY']
gl_api_key_1 = 1

bootstrap = Bootstrap(app)
#app.config['MAX_CONTENT_LENGTH'] = 16 * 1000 * 1000

from logging.config import dictConfig

dictConfig(
    {
        "version": 1,
        "formatters": {
            "default": {
                "format": "[%(asctime)s] %(levelname)s in %(module)s: %(message)s",
            }
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stdout",
                "formatter": "default",
            },
            "file": {
                "class": "logging.FileHandler",
                "filename": "py_log.log",
                "formatter": "default",
            },
        },
        "root": {"level": "INFO", "handlers": ["console", "file"]},
    }
)


from app import routes, models