import os
from dotenv import load_dotenv
load_dotenv()
class Config(object):
    SECRET_KEY = os.getenv("SECRET_KEY")
    SECURITY_PASSWORD_SALT =  os.getenv("SECURITY_PASSWORD_SALT")
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    WTF_CSRF_ENABLED = False
    SECURITY_TOKEN_AUTHENTICATION_HEADER = 'Authentication-Token'

class DevelopmentConfig(Config):
    DEBUG = True