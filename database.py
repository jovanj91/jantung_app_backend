import os
from sqlalchemy import create_engine, URL
from sqlalchemy.orm import scoped_session, sessionmaker
from sqlalchemy.ext.declarative import declarative_base

url_object = URL.create(
    "mysql+pymysql",
    username=os.environ.get("DB_USERNAME"),
    password=os.environ.get("DB_PASSWORD"),
    host=os.environ.get("DB_HOST"),
    database=os.environ.get("DB_NAME"),
)
engine = create_engine(url_object)
db_session = scoped_session(sessionmaker(autocommit=False,
                                         autoflush=False,
                                         bind=engine))
Base = declarative_base()
Base.query = db_session.query_property()

def init_db():
    import models
    Base.metadata.create_all(bind=engine)