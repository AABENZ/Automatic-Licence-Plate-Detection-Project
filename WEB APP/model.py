import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img , img_to_array
import pytesseract as pt

model = tf.keras.models.load_model('./static/models/object_detection_model.h5')

def object_detection_model(path,filename):
    image = load_img(path)
    image = np.array(image,dtype=np.uint8) #8 bit array
    image1 = load_img(path,target_size=(224,224))
    img_array = img_to_array(image1)/255.0 #convert into array and get normalized utput
    h,w,d = image.shape
    test_arr = img_array.reshape(1,224,224,3) #nb_img , height , width , dense
    #make prediction
    coords = model.predict(test_arr)
    #denormalize the output
    denorm = np.array([w,w,h,h])
    coords = coords * denorm
    #convert values to int
    coords = coords.astype(np.int32)
    #draw boundingbox on the image
    # we need two diagonal points
    xmin,xmax,ymin,ymax = coords[0] #get the values of first row
    pt1 = (xmin,ymin)
    pt2 = (xmax,ymax)
    #print the diagonal points
    print(pt1,pt2)
    #draw the boundingbox
    cv2.rectangle(image,pt1,pt2,(0,255,0),3) #image in rgb format
    #coner image to bgr format
    img_bgr = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
    cv2.imwrite('./static/predict/{}'.format(filename),img_bgr)
    return coords

def OCR(path,filename):
    img = np.array(load_img(path))
    coords = object_detection_model(path,filename)
    #crop image to get only the bounding box
    xmin ,xmax,ymin,ymax = coords[0]
    # the bounding box
    roi = img[ymin:ymax,xmin:xmax]
    roi_bgr = cv2.cvtColor(roi,cv2.COLOR_RGB2BGR)
    cv2.imwrite('./static/roi/{}'.format(filename),roi_bgr)
    text = pt.image_to_string(roi)
    print(text)
    return text