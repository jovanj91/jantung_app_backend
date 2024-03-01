from PyQt5 import QtWidgets,uic
from PyQt5 import QtGui
from PyQt5 import QtCore
from PyQt5.QtWidgets import QMainWindow, QApplication, QLabel, QFileDialog, QAction
from PyQt5.QtGui import QPixmap
import cv2
import subprocess
import win32api
index = 0

def load_image_PSAX(self):
    global index
    index = 1
    directory = str(QFileDialog.getExistingDirectory())
    imagePath1 = directory + "/PIC*.jpg"
    Txtfile = open("Reportpsax.txt", "w")
    Txtfile.write(imagePath1)
    Txtfile.close()
def process_PSAX(self):
	subprocess.call(['C:\\Users\\Anwar\\source\\repos\\MULTIDIMENSI\\SHORTAXIS\\x64\\Debug\\SHORTAXIS.exe'])
def tracking_PSAX(self):
    imagePath = "Trackingpsax.jpg"
    pixmap = QPixmap(imagePath)
    pixmap = pixmap.scaled(301,201)
    call.label.setPixmap(pixmap)
def klasifikasi_PSAX(self):
    result = 1
    if result == 0:
        print("Tidak Normal")
        call.label_9.setText("Tidak Normal")
    else:
        print("Normal")
        call.label_9.setText("Normal")


def load_image_4AC(self):
    global index
    index = 1
    directory = str(QFileDialog.getExistingDirectory())
    imagePath1 = directory + "/PIC*.jpg"
    Txtfile = open("Report4ac.txt", "w")
    Txtfile.write(imagePath1)
    Txtfile.close()
def process_4AC(self):
	subprocess.call(['C:\\Users\\Anwar\\source\\repos\\MULTIDIMENSI\\FOURCHAMBER\\x64\\Debug\\FOURCHAMBER.exe'])
def tracking_4AC(self):
    imagePath = "Tracking4ac.jpg"
    pixmap = QPixmap(imagePath)
    pixmap = pixmap.scaled(301,201)
    call.label_3.setPixmap(pixmap)
def klasifikasi_4AC(self):
    result = 1
    if result == 0:
        print("Tidak Normal")
        call.label_11.setText("Tidak Normal")
    else:
        print("Normal")
        call.label_11.setText("Normal")


def load_image_2AC(self):
    global index
    index = 1
    directory = str(QFileDialog.getExistingDirectory())
    imagePath1 = directory + "/PIC*.jpg"
    Txtfile = open("Report2ac.txt", "w")
    Txtfile.write(imagePath1)
    Txtfile.close()
def process_2AC(self):
	subprocess.call(['C:\\Users\\Anwar\\source\\repos\\MULTIDIMENSI\\TWOCHAMBER\\x64\\Debug\\TWOCHAMBER.exe'])
def tracking_2AC(self):
    imagePath = "Tracking2ac.jpg"
    pixmap = QPixmap(imagePath)
    pixmap = pixmap.scaled(301,201)
    call.label_4.setPixmap(pixmap)
def klasifikasi_2AC(self):
    result = 1
    if result == 0:
        print("Tidak Normal")
        call.label_12.setText("Tidak Normal")
    else:
        print("Normal")
        call.label_12.setText("Normal")

  
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
