from PyQt5 import QtWidgets,uic
from PyQt5 import QtGui
from PyQt5 import QtCore
from PyQt5.QtWidgets import QMainWindow, QApplication, QLabel, QFileDialog, QAction
from PyQt5.QtGui import QPixmap
from time import sleep
import subprocess
import win32api
import pickle
from sklearn.externals import joblib
from sklearn import preprocessing
import numpy as np
import pandas as pd
index = 0
index1 = 0
index2 = 0

####################### SHORT AXIS PROCESS (0-4) #########################
#LoadImage
def load_image_PSAX():
    global index
    if index == 0 or index == 1 or index == 4:
        index = 1
        directory = str(QFileDialog.getExistingDirectory())
        imagePath1 = directory + "/PIC*.jpg"
        Txtfile = open("Reportpsax.txt", "w")
        Txtfile.write(imagePath1)
        Txtfile.close()
        call.label_9.setText("RESULT")
    elif index == 2:
        print("Pilih Tracking")
        win32api.MessageBox(0, 'Pilih Tracking', 'Pemberitahuan PSAX')
    else:
        print("Pilih Klasifikasi")
        win32api.MessageBox(0, 'Pilih Klasifikasi', 'Pemberitahuan PSAX')

#Process
def process_PSAX():
    global index
    if index == 1:
        index = 2
        subprocess.call(['C:\\Users\\Anwar\\source\\repos\\MULTIDIMENSI\\SHORTAXIS\\x64\\Debug\\SHORTAXIS.exe'])
        sleep(0.1)
        imagePath = "CitraAslipsax.jpg"
        pixmap = QPixmap(imagePath)
        pixmap = pixmap.scaled(301,201)
        call.label.setPixmap(pixmap)
    elif index == 2:
        print("Pilih Tracking")
        win32api.MessageBox(0, 'Pilih Tracking', 'Pemberitahuan PSAX')
    elif index == 3:
        print("Pilih Klasifikasi")
        win32api.MessageBox(0, 'Pilih Klasifikasi', 'Pemberitahuan PSAX')
    else:
        print("Pilih Load Data")
        win32api.MessageBox(0, 'Pilih Load Data', 'Pemberitahuan PSAX')


#Tracking
def tracking_PSAX():
    global index
    if index == 2:
        index = 3
        imagePath = "Trackingpsax.jpg"
        pixmap = QPixmap(imagePath)
        pixmap = pixmap.scaled(301,201)
        call.label.setPixmap(pixmap)
    elif index == 0:
        print("Pilih Load Data")
        win32api.MessageBox(0, 'Pilih Load Data', 'Pemberitahuan PSAX')
    elif index == 1:
        print("Pilih Process")
        win32api.MessageBox(0, 'Pilih Process', 'Pemberitahuan PSAX')
    elif index == 3:
        print("Pilih Klasifikasi")
        win32api.MessageBox(0, 'Pilih Klasifikasi', 'Pemberitahuan PSAX')
    else:
        print("Pilih Load Data")
        win32api.MessageBox(0, 'Pilih Load Data', 'Pemberitahuan PSAX')

#Klasifikasi
def klasifikasi_PSAX():
    global index
    if index == 3:

        df = pd.read_csv('M1F1_PSAX.csv')
        X = df.drop('CLASS', axis=1)
        X = preprocessing.StandardScaler().fit(X).transform(X.astype(float))
        temp = X.shape[0]
        filename = 'SVM_PSAX'
        loaded_model = joblib.load(filename)
        model = pickle.dumps(loaded_model)
        prediction = pickle.loads(model)
        result = prediction.predict(X[temp-1:temp])
        print(result)

        with open("M1F1_PSAX.csv", "r") as data:
            lines = data.readlines()
            lines = lines[:-1]
        with open("M1F1_PSAX.csv", "w") as data:
            for line in lines:
                data.write(line)
        
        if result == 0:
            print("Tidak Normal")
            call.label_9.setText("Tidak Normal")
        else:
            print("Normal")
            call.label_9.setText("Normal")
        index = 4
    elif index == 1:
        print("Pilih Process")
        win32api.MessageBox(0, 'Pilih Process', 'Pemberitahuan PSAX')
    elif index == 2:
        print("Pilih Tracking")
        win32api.MessageBox(0, 'Pilih Tracking', 'Pemberitahuan PSAX')
    elif index == 4 or index == 0:
        print("Pilih Load Data")
        win32api.MessageBox(0, 'Pilih Load Data', 'Pemberitahuan PSAX')

####################### SHORT AXIS END PROCESS ######################### 


####################### FOUR CHAMBER PROCESS (0-4) #########################
#LoadImage
def load_image_4AC():
    global index1
    if index1 == 0 or index1 == 1 or index1 == 4:
        index1 = 1
        directory = str(QFileDialog.getExistingDirectory())
        imagePath1 = directory + "/PIC*.jpg"
        Txtfile = open("Report4ac.txt", "w")
        Txtfile.write(imagePath1)
        Txtfile.close()
        call.label_11.setText("RESULT")
    elif index1 == 2:
        print("Pilih Tracking")
        win32api.MessageBox(0, 'Pilih Tracking', 'Pemberitahuan 4AC')
    else:
        print("Pilih Klasifikasi")
        win32api.MessageBox(0, 'Pilih Klasifikasi', 'Pemberitahuan 4AC')

#Process
def process_4AC():
    global index1
    if index1 == 1:
        index1 = 2
        subprocess.call(['C:\\Users\\Anwar\\source\\repos\\MULTIDIMENSI\\FOURCHAMBER\\x64\\Debug\\FOURCHAMBER.exe'])
        sleep(0.1)
        imagePath = "CitraAsli4ac.jpg"
        pixmap = QPixmap(imagePath)
        pixmap = pixmap.scaled(301,201)
        call.label_3.setPixmap(pixmap)
    elif index1 == 2:
        print("Pilih Tracking")
        win32api.MessageBox(0, 'Pilih Tracking', 'Pemberitahuan 4AC')
    elif index1 == 3:
        print("Pilih Klasifikasi")
        win32api.MessageBox(0, 'Pilih Klasifikasi', 'Pemberitahuan 4AC')
    else:
        print("Pilih Load Data")
        win32api.MessageBox(0, 'Pilih Load Data', 'Pemberitahuan 4AC')


#Tracking
def tracking_4AC():
    global index1
    if index1 == 2:
        index1 = 3
        imagePath = "Tracking4ac.jpg"
        pixmap = QPixmap(imagePath)
        pixmap = pixmap.scaled(301,201)
        call.label_3.setPixmap(pixmap)
    elif index1 == 0:
        print("Pilih Load Data")
        win32api.MessageBox(0, 'Pilih Load Data', 'Pemberitahuan 4AC')
    elif index1 == 1:
        print("Pilih Process")
        win32api.MessageBox(0, 'Pilih Process', 'Pemberitahuan 4AC')
    elif index1 == 3:
        print("Pilih Klasifikasi")
        win32api.MessageBox(0, 'Pilih Klasifikasi', 'Pemberitahuan 4AC')
    else:
        print("Pilih Load Data")
        win32api.MessageBox(0, 'Pilih Load Data', 'Pemberitahuan 4AC')

#Klasifikasi
def klasifikasi_4AC():
    global index1
    if index1 == 3:

        df = pd.read_csv('M1F1_4AC.csv')
        X = df.drop('CLASS', axis=1)
        X = preprocessing.StandardScaler().fit(X).transform(X.astype(float))
        temp = X.shape[0]
        filename = 'SVM_4AC'
        loaded_model = joblib.load(filename)
        model = pickle.dumps(loaded_model)
        prediction = pickle.loads(model)
        result = prediction.predict(X[temp-1:temp])
        print(result)

        with open("M1F1_4AC.csv", "r") as data:
            lines = data.readlines()
            lines = lines[:-1]
        with open("M1F1_4AC.csv", "w") as data:
            for line in lines:
                data.write(line)
        
        if result == 0:
            print("Tidak Normal")
            call.label_11.setText("Tidak Normal")
        else:
            print("Normal")
            call.label_11.setText("Normal")
        index1 = 4
    elif index1 == 1:
        print("Pilih Process")
        win32api.MessageBox(0, 'Pilih Process', 'Pemberitahuan 4AC')
    elif index1 == 2:
        print("Pilih Tracking")
        win32api.MessageBox(0, 'Pilih Tracking', 'Pemberitahuan 4AC')
    elif index1 == 4 or index1 == 0:
        print("Pilih Load Data")
        win32api.MessageBox(0, 'Pilih Load Data', 'Pemberitahuan 4AC')

####################### FOUR CHAMBER END PROCESS ######################### 

####################### TWO CHAMBER PROCESS (0-4) #########################
#LoadImage
def load_image_2AC():
    global index2
    if index2 == 0 or index2 == 1 or index2 == 4:
        index2 = 1
        directory = str(QFileDialog.getExistingDirectory())
        imagePath1 = directory + "/PIC*.jpg"
        Txtfile = open("Report2ac.txt", "w")
        Txtfile.write(imagePath1)
        Txtfile.close()
        call.label_12.setText("RESULT")
    elif index2 == 2:
        print("Pilih Tracking")
        win32api.MessageBox(0, 'Pilih Tracking', 'Pemberitahuan 2AC')
    else:
        print("Pilih Klasifikasi")
        win32api.MessageBox(0, 'Pilih Klasifikasi', 'Pemberitahuan 2AC')

#Process
def process_2AC():
    global index2
    if index2 == 1:
        index2 = 2
        subprocess.call(['C:\\Users\\Anwar\\source\\repos\\MULTIDIMENSI\\TWOCHAMBER\\x64\\Debug\\TWOCHAMBER.exe'])
        sleep(0.1)
        imagePath = "CitraAsli2ac.jpg"
        pixmap = QPixmap(imagePath)
        pixmap = pixmap.scaled(301,201)
        call.label_4.setPixmap(pixmap)
    elif index2 == 2:
        print("Pilih Tracking")
        win32api.MessageBox(0, 'Pilih Tracking', 'Pemberitahuan 2AC')
    elif index2 == 3:
        print("Pilih Klasifikasi")
        win32api.MessageBox(0, 'Pilih Klasifikasi', 'Pemberitahuan 2AC')
    else:
        print("Pilih Load Data")
        win32api.MessageBox(0, 'Pilih Load Data', 'Pemberitahuan 2AC')


#Tracking
def tracking_2AC():
    global index2
    if index2 == 2:
        index2 = 3
        imagePath = "Tracking2ac.jpg"
        pixmap = QPixmap(imagePath)
        pixmap = pixmap.scaled(301,201)
        call.label_4.setPixmap(pixmap)
    elif index2 == 0:
        print("Pilih Load Data")
        win32api.MessageBox(0, 'Pilih Load Data', 'Pemberitahuan 2AC')
    elif index2 == 1:
        print("Pilih Process")
        win32api.MessageBox(0, 'Pilih Process', 'Pemberitahuan 2AC')
    elif index2 == 3:
        print("Pilih Klasifikasi")
        win32api.MessageBox(0, 'Pilih Klasifikasi', 'Pemberitahuan 2AC')
    else:
        print("Pilih Load Data")
        win32api.MessageBox(0, 'Pilih Load Data', 'Pemberitahuan 2AC')

#Klasifikasi
def klasifikasi_2AC():
    global index2
    if index2 == 3:

        df = pd.read_csv('M1F1_2AC.csv')
        X = df.drop('CLASS', axis=1)
        X = preprocessing.StandardScaler().fit(X).transform(X.astype(float))
        temp = X.shape[0]
        filename = 'SVM_2AC'
        loaded_model = joblib.load(filename)
        model = pickle.dumps(loaded_model)
        prediction = pickle.loads(model)
        result = prediction.predict(X[temp-1:temp])
        print(result)

        with open("M1F1_2AC.csv", "r") as data:
            lines = data.readlines()
            lines = lines[:-1]
        with open("M1F1_2AC.csv", "w") as data:
            for line in lines:
                data.write(line)
        
        if result == 0:
            print("Tidak Normal")
            call.label_12.setText("Tidak Normal")
        else:
            print("Normal")
            call.label_12.setText("Normal")
        index2 = 4
    elif index2 == 1:
        print("Pilih Process")
        win32api.MessageBox(0, 'Pilih Process', 'Pemberitahuan 2AC')
    elif index2 == 2:
        print("Pilih Tracking")
        win32api.MessageBox(0, 'Pilih Tracking', 'Pemberitahuan 2AC')
    elif index2 == 4 or index2 == 0:
        print("Pilih Load Data")
        win32api.MessageBox(0, 'Pilih Load Data', 'Pemberitahuan 2AC')

####################### TWO CHAMBER END PROCESS ######################### 


  
#setup gui
app=QtWidgets.QApplication([])
call=uic.loadUi("multiview.ui")

#SHORTAXIS PROCESS
#button callback function load video
call.pushButton.clicked.connect(load_image_PSAX)
#button callback function median
call.pushButton_1.clicked.connect(process_PSAX)
#button callback function highboost
call.pushButton_2.clicked.connect(tracking_PSAX)
#button callback function klasifikasi
call.pushButton_12.clicked.connect(klasifikasi_PSAX)

#LONGAXIS PROCESS
#button callback function load video
#call.pushButton_3.clicked.connect(load_image_PSAX)
#button callback function median
#call.pushButton_4.clicked.connect(process_PSAX)
#button callback function highboost
#call.pushButton_5.clicked.connect(tracking_PSAX)

#FOURCHAMBER PROCESS
#button callback function load video
call.pushButton_6.clicked.connect(load_image_4AC)
#button callback function median
call.pushButton_7.clicked.connect(process_4AC)
#button callback function highboost
call.pushButton_8.clicked.connect(tracking_4AC)
#button callback function klasifikasi
call.pushButton_14.clicked.connect(klasifikasi_4AC)

#TWOCHAMBER PROCESS
#button callback function load video
call.pushButton_9.clicked.connect(load_image_2AC)
#button callback function median
call.pushButton_10.clicked.connect(process_2AC)
#button callback function highboost
call.pushButton_11.clicked.connect(tracking_2AC)
#button callback function klasifikasi
call.pushButton_15.clicked.connect(klasifikasi_2AC)

#show and execute
call.show()
app.exec()
