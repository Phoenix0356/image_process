import cv2
import numpy as np

class hightpass_filter():

    def apply_filter(self, img, filter):
        return cv2.filter2D(img, -1, filter)
    def roborts(self,img):
        return self.apply_filter(img,np.array([[1, 0], [0, -1]]))
        # return cv2.filter2D(img, -1, np.array([[1, 0], [0, -1]]))

    def prewitts(self,img):
        prewitt_x = self.apply_filter(img,np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]))
        prewitt_y = self.apply_filter(img,np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]]))
        return abs(prewitt_x) + abs(prewitt_y)

    def sobel(self,img):
        sobel_x = cv2.Sobel(img, cv2.CV_64F, 2, 0, ksize=5)
        sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 2, ksize=5)
        return abs(sobel_x)+abs(sobel_y)

    def laplace(self,img):
        return cv2.Laplacian(img,cv2.CAP_VFW)

    def canny(self,img):
        return cv2.Canny(img,50,250)