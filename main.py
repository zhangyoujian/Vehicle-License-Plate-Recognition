import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
ROOT_DIR='./'
sys.path.append(ROOT_DIR)

from QtUi.MainWindow import mainWindow
from PyQt5.QtWidgets import QApplication, QWidget


def main():
    app = QApplication(sys.argv)
    w = mainWindow()
    w.show()
    sys.exit(app.exec_())


if __name__=='__main__':
   main()


