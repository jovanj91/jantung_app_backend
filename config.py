import os
from dotenv import load_dotenv

load_dotenv()
class Config(object):
    SECRET_KEY = os.getenv("SECRET_KEY", 'pf9Wkove4IKEAXvy-cQkeDPhv9Cb3Ag-wyJILbq_dFw')
    SECURITY_PASSWORD_SALT =  os.getenv("SECURITY_PASSWORD_SALT", '146585145368132386173505678016728509634')
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    WTF_CSRF_ENABLED = False
    SECURITY_TOKEN_AUTHENTICATION_HEADER = 'Authentication-Token'

class DevelopmentConfig(Config):
    DEBUG = True