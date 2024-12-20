import app
from app import app
from app.models import Files
from openai import OpenAI
import openai
from app import gl_api_key, gl_api_key_1


def check(api):
    # print('api_key',api_key)
    print('gl_api_key_1', api)
    return True

def set1(u_api_key):
    global gl_api_key_1
    gl_api_key_1 = u_api_key
    return




