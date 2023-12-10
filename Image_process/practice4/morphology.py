import cv2
import numpy as np

class morphology():
    def erode_operation(self,img,kernel,iteration):
        return cv2.erode(img, kernel, iterations=iteration)
    def dilate_operation(self,img,kernel,iteration):
        return cv2.dilate(img, kernel, iterations=iteration)

