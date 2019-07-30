import os
import sys
import time
import cv2
import sys
import numpy as np
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import (QWidget,QMainWindow,QFileDialog,QApplication)
from PyQt5.QtGui import QPixmap,QIcon,QPicture,QImage
from PyQt5.QtCore import QObject,QThread
from threading import *
from CharSplit.CharSplit import CharSplit
from QtUi.LoadMaskRCNNModel import MaskRCNN as MaskRCNN
from QtUi.LoadMaskRCNNModel import random_colors
from QtUi.MainWindowUi import Ui_MainWindow as mainWindowObj
import copy
from skimage.transform import rotate, resize
from DigitRecognize import CNN


class mainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.UI = mainWindowObj()
        self.UI.setupUi(self)
        self.setFixedSize(self.size())
        self.UI.labelImage.setScaledContents(True)
        self.statusBar().showMessage('No Images Display')
        self.UI.btnOk.clicked.connect(self.on_slot_btnOk)
        self.UI.btnCancle.clicked.connect(self.on_slot_btnCancle)
        self.UI.btnExit.clicked.connect(self.on_slot_btnExit)
        self.UI.action_Open.triggered.connect(self.on_slot_Open)
        self.UI.action_Exit.triggered.connect(self.close)
        self.UI.btnCharSplit.clicked.connect(self.on_charSplit)

        self.pixmap = QPixmap()
        self.model = MaskRCNN()
        self.Processimage = None
        self.LicensePlate =[]
        self.chineseModel = CNN.CNNChinese((32, 32, 1))
        self.chineseModel.LoadModel('./DigitRecognize/OCR_model_weight.h5')
        self.letterModel = CNN.CNNLetter((28, 28, 1))
        self.letterModel.LoadModel('./DigitRecognize/OCR_model_letters_weight.h5')


    def closeEvent(self, event):
        print('主窗口已关闭!')
        self.close()



    def on_slot_btnOk(self):
        if self.Processimage is None:
            return
        ROIS,MASK = self.model.MaskRCNNDetect(self.Processimage)
        N = ROIS.shape[0]
        if not N:
            print('没有检测到车牌')
            return
        else:
            print('检测到有%d个车牌'%(N))

        markImages = copy.deepcopy(self.Processimage)
        color = random_colors(N)
        License = []
        for i in range(N):
            startX = ROIS[i, 1]
            startY = ROIS[i, 0]
            width = ROIS[i, 2] - startY
            height = ROIS[i, 3] - startX
            temp = markImages[startY:startY + width, startX:startX + height, :]
            License.append(temp)

            markImages = cv2.rectangle(markImages, (
                startX - 2, startY - 2, height + 4, width + 10), color[i], 3)
            plt.subplots(1+i, figsize=(0.6, 1.5))
            plt.imshow(temp.astype(np.uint8))

        plt.figure(N+1, figsize=((6.5, 6.5)))
        plt.imshow(markImages.astype(np.uint8))
        plt.show()
        self.LicensePlate = License


    def on_slot_btnCancle(self):
        print('取消')

    def on_slot_btnExit(self):
        self.close()

    def on_slot_Open(self):
        filter = "图片文件(*.jpg *.gif *.png *.jpeg *.bmp *.tiff)"
        fileName = QFileDialog.getOpenFileName(self, 'Select a Pictrue', './', filter)

        if fileName[0]:
            str = fileName[0].split('/')
            self.statusBar().showMessage(str[-1])
            image = self.model.ReadImage(fileName[0])
            self.Processimage = image
            self.pixmap.load(fileName[0])
            width = self.UI.labelImage.width()
            height = self.UI.labelImage.height()
            pixmap = self.pixmap.scaled(width, height)
            self.UI.labelImage.setPixmap(pixmap)


    def on_charSplit(self):
        if len(self.LicensePlate)<1:
            return

        for Image in self.LicensePlate:
            # 颜色转换
            image_gray = cv2.cvtColor(Image, cv2.COLOR_BGR2GRAY)

            # 计算倾斜角度并做相应的旋转
            angle = CharSplit.randonTransform(image_gray)
            print('旋转角度:', angle)
            image_gray_rotate = CharSplit.imrotate(image_gray, angle)

            plt.subplot(1, 2, 1), plt.imshow(image_gray, 'gray')
            plt.title('original')
            plt.subplot(1, 2, 2), plt.imshow(image_gray_rotate, 'gray')
            plt.title('rotate')
            plt.show()

            # 将灰度图像转换为二进制图像
            image_open = CharSplit.BinaryImage(image_gray_rotate)

            # 对二值化后的图像进行水平投影分析
            markrow, markrow1, markrow2= CharSplit.LevelRowAnalyse(image_open)
            sbw2, maxhight, rowtop, rowbot = CharSplit.rowAnalyse(image_open, markrow, markrow1, markrow2)

#           垂直投影分析
            retGrayImage, retBinayImage = CharSplit.colAnalyse(image_gray_rotate, image_open, sbw2, maxhight, rowtop,rowbot)
            self.PredictPlate(retGrayImage,retBinayImage)


    def PredictPlate(self,ImageList,BinarImageList):
        if len(ImageList)<=0:
            return

        chineseImage = ImageList[0]
        if np.max(chineseImage)>1:
            chineseImage = chineseImage/255.0

        chineseImage = resize(chineseImage, (32, 32))
        chineseImage = np.array([chineseImage])

        letterImageList = []
        for k in range(1,len(ImageList)):
            if np.max(BinarImageList[k])>1:
                ImageList[k] = BinarImageList[k]/255.0
            letterImageList.append(BinarImageList[k])

        letterImage = [resize(image, (28, 28)) for image in letterImageList]

        letterImage = np.array(letterImage)


        Ch1 = self.chineseModel.predict(chineseImage)
        Ch2 = self.letterModel.predict(letterImage)
        result = Ch1+Ch2
        print("识别结果:"+str(result))













