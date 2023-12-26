from database import Base
from flask_security import UserMixin, RoleMixin, AsaList
from sqlalchemy.orm import relationship, backref
from sqlalchemy.ext.mutable import MutableList
from sqlalchemy import Boolean, DateTime, Column, Integer, \
                    String, ForeignKey, Double, Date, func

class RolesUsers(Base):
    __tablename__ = 'roles_users'
    id = Column(Integer(), primary_key=True)
    user_id = Column('user_id', Integer(), ForeignKey('user.id'))
    role_id = Column('role_id', Integer(), ForeignKey('role.id'))

class Role(Base, RoleMixin):
    __tablename__ = 'role'
    id = Column(Integer(), primary_key=True)
    name = Column(String(80), unique=True)
    description = Column(String(255))
    permissions = Column(MutableList.as_mutable(AsaList()), nullable=True)

class User(Base, UserMixin):
    __tablename__ = 'user'
    id = Column(Integer(), primary_key=True)
    email = Column(String(255), unique=True)
    username = Column(String(255), unique=True, nullable=True)
    password = Column(String(255), nullable=False)
    last_login_at = Column(DateTime())
    current_login_at = Column(DateTime())
    last_login_ip = Column(String(100))
    current_login_ip = Column(String(100))
    login_count = Column(Integer)
    active = Column(Boolean())
    fs_uniquifier = Column(String(64), unique=True, nullable=False)
    confirmed_at = Column(DateTime())
    roles = relationship('Role', secondary='roles_users', backref=backref('users', lazy='dynamic'))
    patient_data = relationship('PatientData', backref='user', lazy='dynamic')
    def __repr__(self):
        return f"<User(id={self.id}, username='{self.username}', email='{self.email}')>"

class PatientData(Base):
    __tablename__ = 'patientData'
    id = Column(Integer(), primary_key=True)
    user_id = Column('user_id', Integer(), ForeignKey('user.id'))
    patient_name = Column(String(128))
    dob = Column(Date())
    gender = Column(Integer())
    heart_check = relationship('HeartCheck', backref='patient', lazy='dynamic')

class HeartCheck(Base):
    __tablename__ = 'heartCheck'
    id = Column(Integer(), primary_key=True)
    patient_id = Column('patient_id', Integer(), ForeignKey('patientData.id'))
    age = Column(Integer())
    checkResult = Column(String(24))
    video_path = Column(String(255))
    checked_at = Column(DateTime())