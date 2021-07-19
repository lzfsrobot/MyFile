#-*- coding: utf-8 -*-

import cv2.cv2 as cv2
import sys
import gc
import tensorflow as tf
import load_data
import numpy as np
from load_data import load_dataset
from sklearn.model_selection import train_test_split

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
import random
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)


name = ['lzf', 'yeye', 'houxiaozi']
def CatchPICFromVideo(window_name):
    cv2.namedWindow(window_name)
    model = tf.keras.models.load_model(r'cnn_network.h5')    
    #视频来源，可以来自一段已存好的视频，也可以直接来自USB摄像头
    cap = cv2.VideoCapture(0)                
    
    #告诉OpenCV使用人脸识别分类器
    classfier = cv2.CascadeClassifier(r"D:\Manim\scoop\apps\anaconda3\2020.07\Lib\site-packages\cv2\data\haarcascade_frontalface_alt2.xml")
    
    #识别出人脸后要画的边框的颜色，RGB格式
    color = (0, 255, 0)
    num = 0
    while cap.isOpened():
        ok, frame = cap.read() #读取一帧数据
        if not ok:            
            break        

        frame = cv2.flip(frame, 1)
        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  #将当前桢图像转换成灰度图像            
        
        #人脸检测，1.2和2分别为图片缩放比例和需要检测的有效点数
        faceRects = classfier.detectMultiScale(grey, scaleFactor = 1.2, minNeighbors = 3, minSize = (32, 32))
        if len(faceRects) > 0:          #大于0则检测到人脸                                   
            for faceRect in faceRects:  #单独框出每一张人脸
                x, y, w, h = faceRect                        
                num += 1
                image = frame[y - 10: y + h + 10, x - 10: x + w + 10] 
                image = load_data.resize_image(image)
                image = image[np.newaxis, ...]

                faceID = model.predict(image)   
                faceID = tf.argmax(faceID, axis=1)                               
                # print(faceID)
                faceID = int(faceID)
                #画出矩形框
                print(faceID)
                cv2.rectangle(frame, (x - 10, y - 10), (x + w + 10, y + h + 10), color, 2)
                                                                       
                cv2.rectangle(frame, (x - 10, y - 10), (x + w + 10, y + h + 10), color, thickness = 2)
                
                #文字提示是谁
                cv2.putText(frame,name[faceID], 
                            (x + 30, y + 30),                      #坐标
                            cv2.FONT_HERSHEY_SIMPLEX,              #字体
                            1,                                     #字号
                            (255, 0, 255),                           #颜色
                            2)                                     #字的线宽       
                       
                       
        #显示图像
        cv2.imshow(window_name, frame)        
        c = cv2.waitKey(10)
        if c & 0xFF == ord('q'):
            break
    
    #释放摄像头并销毁所有窗口
    cap.release()
    cv2.destroyAllWindows()

CatchPICFromVideo("person_reg")



