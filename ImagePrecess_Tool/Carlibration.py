#coding=utf-8
import numpy as np
import cv2
import os
import time
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD

import keras
import numpy as np

class Auto_calibrating():
    def __init__(self):
        pass

    def norm(self, X):
        # X = cv2.resize(X,(28,28))
        P = cv2.cvtColor(X, cv2.COLOR_RGB2GRAY)
        P = P.astype(np.float32) / 255
        P = np.expand_dims(P, 2)
        return P

    def rectSrceen(self, pts, rect):
        zeroadd = np.array(rect[:2])
        return np.array(pts) + zeroadd

    def getTransformMatrix(self,origin_shape, shape_=(140, 180)):
        pts1 = np.float32([[0, 0], [origin_shape[1], 0], [origin_shape[1], origin_shape[0]], [0, origin_shape[0]]])
        pts2 = np.float32([[0, 0], [shape_[1], 0], [shape_[1], shape_[0]], [0, shape_[0]]])
        M = cv2.getPerspectiveTransform(pts1, pts2)
        return M

    def resize_with_points(self,pts, shape, image):
        M = self.getTransformMatrix(image.shape, shape)
        points_ = np.dot(M, np.float32(np.vstack((pts.T, np.array([1, 1, 1, 1]).T))))[0:2, 0:4].T.astype(np.int)
        return points_

    def foutpoint(self,p_points):
        shape = [140, 180]
        p_points = p_points[0]
        xp = np.array([[p_points[0] * shape[1], p_points[1] * shape[0]],
                       [p_points[2] * shape[1], p_points[3] * shape[0]],
                       [p_points[4] * shape[1], p_points[5] * shape[0]],
                       [p_points[6] * shape[1], p_points[7] * shape[0]]
                       ])
        return xp

    def drawPoints(img, p_points):
        shape = [140, 180]
        p_points = p_points[0]
        p = np.array([[p_points[0] * shape[1], p_points[1] * shape[0]],
                      [p_points[2] * shape[1], p_points[3] * shape[0]],
                      [p_points[4] * shape[1], p_points[5] * shape[0]],
                      [p_points[6] * shape[1], p_points[7] * shape[0]]
                      ])
        for one in p:
            cv2.circle(img, (int(one[0]), int(one[1])), 1, (0, 255, 0), 2)
        cv2.imshow("img", img)
        cv2.waitKey(0)

    def getmodel(self, path):
        model = Sequential()
        model.add(Convolution2D(1, 7, 7, border_mode='valid', input_shape=(140, 180, 1), subsample=(2, 2)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(3, 3)))
        model.add(Convolution2D(1, 3, 3, border_mode='valid', subsample=(2, 2)))
        model.add(Convolution2D(32, 5, 5, border_mode='valid', subsample=(2, 2)))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Activation('relu'))
        # model.add(Dropout(0.1))

        model.add(Flatten())
        model.add(Dense(128, init='normal'))
        model.add(Activation('relu'))
        model.add(Dense(8, init='normal'))
        model.add(Activation('relu'))
        model.load_weights(path)
        return model


    def Precess(self,image):

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        extend_patch = gray
        extend_patch_resize = cv2.resize(gray, (180, 140))
        models = self.getmodel('model_t5_regression.h5')
        image = self.norm(extend_patch_resize)
        vector = models.predict(np.array([image]))
        points = self.foutpoint(vector)
        points = self.resize_with_points(points, extend_patch.shape, extend_patch_resize)
        # points = self.rectSrceen(points, [start_corr_x, start_corr_y])

        remap_points = np.array([[0, 0], [136, 0], [136, 36], [0, 36]], dtype=np.float32)

        print(points, remap_points)
        M = cv2.getPerspectiveTransform(np.array(points, dtype=np.float32), remap_points)
        warp_image = cv2.warpPerspective(image, M, (136, 36))
        cv2.imshow("warp_image", warp_image)
        for one in points:
            cv2.circle(image, (int(one[0]), int(one[1])), 1, (0, 255, 0), 2)
            # drawPoints(extend_patch, vector)


        cv2.imshow("img", image)
        cv2.waitKey(0)




