# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'MainWindowUi.ui'
#
# Created by: PyQt5 UI code generator 5.11.3
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(793, 593)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.labelImage = QtWidgets.QLabel(self.centralwidget)
        self.labelImage.setMinimumSize(QtCore.QSize(510, 0))
        self.labelImage.setText("")
        self.labelImage.setObjectName("labelImage")
        self.gridLayout_2.addWidget(self.labelImage, 0, 0, 1, 1)
        self.groupBox = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox.setMaximumSize(QtCore.QSize(100, 16777215))
        self.groupBox.setObjectName("groupBox")
        self.gridLayout = QtWidgets.QGridLayout(self.groupBox)
        self.gridLayout.setObjectName("gridLayout")
        self.btnOk = QtWidgets.QPushButton(self.groupBox)
        self.btnOk.setObjectName("btnOk")
        self.gridLayout.addWidget(self.btnOk, 0, 0, 1, 1)
        self.btnCharSplit = QtWidgets.QPushButton(self.groupBox)
        self.btnCharSplit.setObjectName("btnCharSplit")
        self.gridLayout.addWidget(self.btnCharSplit, 1, 0, 1, 1)
        self.btnCancle = QtWidgets.QPushButton(self.groupBox)
        self.btnCancle.setObjectName("btnCancle")
        self.gridLayout.addWidget(self.btnCancle, 2, 0, 1, 1)
        spacerItem = QtWidgets.QSpacerItem(20, 406, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.gridLayout.addItem(spacerItem, 3, 0, 1, 1)
        self.btnExit = QtWidgets.QPushButton(self.groupBox)
        self.btnExit.setObjectName("btnExit")
        self.gridLayout.addWidget(self.btnExit, 4, 0, 1, 1)
        self.gridLayout_2.addWidget(self.groupBox, 0, 1, 1, 1)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 793, 23))
        self.menubar.setObjectName("menubar")
        self.menu_F = QtWidgets.QMenu(self.menubar)
        self.menu_F.setObjectName("menu_F")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.action_Open = QtWidgets.QAction(MainWindow)
        self.action_Open.setObjectName("action_Open")
        self.action_Exit = QtWidgets.QAction(MainWindow)
        self.action_Exit.setObjectName("action_Exit")
        self.menu_F.addAction(self.action_Open)
        self.menu_F.addSeparator()
        self.menu_F.addAction(self.action_Exit)
        self.menubar.addAction(self.menu_F.menuAction())

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.groupBox.setTitle(_translate("MainWindow", "操作"))
        self.btnOk.setText(_translate("MainWindow", "车牌定位"))
        self.btnCharSplit.setText(_translate("MainWindow", "字符分割"))
        self.btnCancle.setText(_translate("MainWindow", "取消"))
        self.btnExit.setText(_translate("MainWindow", "退出"))
        self.menu_F.setTitle(_translate("MainWindow", "文件(&F)"))
        self.action_Open.setText(_translate("MainWindow", "打开(&O)"))
        self.action_Exit.setText(_translate("MainWindow", "退出(&E)"))

