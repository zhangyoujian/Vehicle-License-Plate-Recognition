import os
import sys
import time
import cv2

from PyQt5.QtWidgets import (QWidget,QMainWindow,QFileDialog)
from PyQt5.QtGui import QPixmap,QIcon,QPicture,QImage
from PyQt5.QtCore import QObject,QThread
from threading import *
from opencv.opencv_tool import *
from UI.mainWindowUi import Ui_UI_MainWindow


class mainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.thstop = False
        self.UI = Ui_UI_MainWindow()
        self.UI.setupUi(self)
        self.setFixedSize(self.size())
        self.UI.btnStart.clicked.connect(self.StartTakePicture)
        self.UI.btnPause.clicked.connect(self.PauseProgress)
        self.UI.OpenAction.triggered.connect(self.LoadPictrue)
        self.UI.btnLocate.clicked.connect(self.locateImage)
        self.UI.labelImage.setScaledContents(True)
        self.setWindowIcon(QIcon('./QIcon/web.jpg'))
        self.Processimage =None
        self.pixmap = QPixmap()
        self.statusBar().showMessage('No Images Display')
        self.show()
    def closeEvent(self, event):
        self.thstop = True
        self.close()
    @staticmethod
    def Mat2QImage(Image):
        curentType=Image.shape[2]
        if curentType==4:
           return QImage(Image.data, Image.shape[1], Image.shape[0],QImage.Format_ARGB32)
        elif curentType==3:
            rgbImage = QImage(Image.data, Image.shape[1], Image.shape[0], QImage.Format_RGB888)
            return rgbImage.rgbSwapped()
        elif curentType==1:
            return QImage(Image.data, Image.shape[1], Image.shape[0],QImage.Format_Grayscale8)
        else:
            return None

    def SetPic(self, img):
        pixmap = QPixmap.fromImage(img)
        width = self.UI.labelImage.width()
        height = self.UI.labelImage.height()
        pixmap = pixmap.scaled(width,height)
        self.UI.labelImage.setPixmap(pixmap)

    def ShowVideoImages(self):
        capture = cv2.VideoCapture(0)
        width = self.UI.labelImage.width()
        height = self.UI.labelImage.height()
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        capture.set(cv2.CAP_PROP_FPS, 30)
        self.statusBar().showMessage('Show Video Images')
        while capture.isOpened():
            if self.thstop:
                return
            ret,frame = capture.read()
            if ret==False:
                continue
            frame = cv2.flip(frame,1)
            a = self.Mat2QImage(frame)
            self.SetPic(a)
            cv2.waitKey(30)
        capture.release()


    def StartTakePicture(self):
        self.thstop=False
        self.ShowVideoImages()

    def PauseProgress(self):
        self.thstop=True

    def LoadPictrue(self):
        filter = "图片文件(*.jpg *.gif *.png *.jpeg *.bmp *.tiff)"
        fileName = QFileDialog.getOpenFileName(self,'Select a Pictrue','./',filter)

        if fileName[0]:
            str = fileName[0].split('/')
            self.statusBar().showMessage(str[-1])
            originalImage = cv_imread(fileName[0])
            self.Processimage = originalImage
            self.pixmap.load(fileName[0])
            width = self.UI.labelImage.width()
            height = self.UI.labelImage.height()
            pixmap = self.pixmap.scaled(width, height)
            self.UI.labelImage.setPixmap(pixmap)


    def locateImage(self):
        getLicencePos(self.Processimage)


