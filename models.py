from sqlalchemy import Column, Integer, String, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import func
# from passlib.hash import bcrypt_sha256
import bcrypt
Base = declarative_base()

class AuthModel(Base):
    __tablename__ = "tb_users"
    user_id = Column(Integer, primary_key = True)
    user_name = Column(String(255))
    user_email = Column(String(255))
    user_password = Column(String(255)) #hashed


    def __init__(self, username, email):
        self.user_name = username
        self.user_email = email

    def set_password(self, password):
        hashedpassword = bcrypt.hashpw(password, bcrypt.gensalt())
        self.user_password = hashedpassword.decode('utf-8')

    def check_password(self, password):
        return bcrypt.checkpw(password, self.user_password.encode('utf-8'))

    def __repr__(self):
        return f"<User(id={self.user_id}, username='{self.user_name}', email='{self.user_email}')>"

