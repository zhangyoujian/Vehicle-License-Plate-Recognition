import numpy as np
import matplotlib.pyplot as plt
import random
import cv2
import copy
import scipy

def cv_imread(filePath):
    cv_img = cv2.imdecode(np.fromfile(filePath, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    # imdecode读取的是rgb，如果后续需要opencv处理的话，需要转换成bgr，转换后图片颜色会变化
    # cv_img = cv2.cvtColor(cv_img, cv2.COLOR_RGB2BGR)
    return cv_img

def getLicencePos(srcImage):
    cv2.setRNGSeed(1234)
    input_image1 = copy.deepcopy(srcImage)
    input_image1 = cv2.cvtColor(input_image1,cv2.COLOR_BGR2GRAY)

    sobelx = np.array([[-0.125,0,0.125],
                       [-0.25,0,0.25],
                       [-0.125,0,0.125]],dtype=np.float32)

    input_image1 = cv2.filter2D(input_image1,cv2.CV_32FC1,sobelx)

    mul_image = cv2.multiply(input_image1,input_image1)

    scaleValue = 4
    threshold = scaleValue * np.mean(mul_image)

    height =input_image1.shape[0]
    width = input_image1.shape[1]
    resultImage = np.zeros(input_image1.shape,dtype=np.float32)
    for i in range(1,height-1):
        for j in range(1,width-1):
            b1 = mul_image[i,j]>mul_image[i,j-1] and mul_image[i,j]>mul_image[i,j+1]
            b2 = mul_image[i,j]>mul_image[i-1,j] and mul_image[i,j]>mul_image[i+1,j]

            resultImage[i,j] = 255*((mul_image[i,j]>threshold) and (b1 or b2))

    resultImage = np.uint8(resultImage)
    cv2.imshow('Binary',resultImage)


    #========================================HSV通道提取======================================================
    input_image2 = copy.deepcopy(srcImage)
    input_image2 = cv2.cvtColor(input_image2,cv2.COLOR_BGR2HSV)
    img_h,img_s,img_v = cv2.split(input_image2)

    img_h = cv2.normalize(np.float32(img_h),None,0,1,cv2.NORM_MINMAX)
    img_s = cv2.normalize(np.float32(img_s), None, 0, 1, cv2.NORM_MINMAX)
    img_v = cv2.normalize(np.float32(img_v), None, 0, 1, cv2.NORM_MINMAX)



    img_vblue1 = img_h>0.45
    img_vblue2 = img_h<0.75
    img_vblue3=img_s>0.15
    img_vblue4 = img_v>0.25

    img_vblue2 = np.bitwise_and(img_vblue1 ,img_vblue2)
    img_vblue3 = np.bitwise_and(img_vblue2,img_vblue3)
    img_vblue = np.bitwise_and(img_vblue3, img_vblue4)


    vbule_gradient = np.bitwise_and(resultImage==255, img_vblue)
    vbule_graientmat = np.zeros(vbule_gradient.shape,dtype=np.uint8)
    vbule_graientmat[vbule_gradient] = 255

    cv2.imshow('Color Merge', vbule_graientmat)

#      形态学提取轮廓

    morph = cv2.morphologyEx(vbule_graientmat,cv2.MORPH_CLOSE,kernel=np.ones((2,25),dtype=np.uint8))
    element = cv2.getStructuringElement(cv2.MORPH_RECT,(11,11))

    morph = cv2.erode(morph,element)

    element = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 17))
    morph =cv2.dilate(morph,element)
    # morph[morph==255]=1
    morph_images = copy.deepcopy(morph)
    contours,hierachy = cv2.findContours(morph,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    color = (0,0,255)

    num=0
    preNum=0


    for oneContours in contours:
        FinalRect = cv2.boundingRect(oneContours)
        temp = morph_images[FinalRect[1]:FinalRect[1]+FinalRect[3], FinalRect[0]:FinalRect[0]+FinalRect[2]]

        num = cv2.countNonZero(temp)
        if num>preNum:
            FinalRectDraw =FinalRect
            preNum =num

    if preNum==0:
        return False

    markImages = cv2.rectangle(srcImage,(FinalRectDraw[0]-5,FinalRectDraw[1]-5,FinalRectDraw[2]+10,FinalRectDraw[3]+10),color,2)
    cv2.imshow('Mark Images',markImages)
    return markImages,FinalRectDraw