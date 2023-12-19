from sqlalchemy import create_engine, URL
from sqlalchemy.orm import scoped_session, sessionmaker
from sqlalchemy.ext.declarative import declarative_base

# engine = create_engine('mysql+pymysql://freedb_admindbjantung:h#j7@#cHBN7Gaf@@sql.freedb.tech:3306/freedb_db_cekjantung')
url_object = URL.create(
    "mysql+pymysql",
    username="freedb_admindbjantung",
    password="h#j7@#cHBN7Gaf@",  # plain (unescaped) text
    host="sql.freedb.tech",
    database="freedb_db_cekjantung",
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