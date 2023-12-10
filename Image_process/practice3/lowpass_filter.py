import cv2
import numpy as np
from scipy.ndimage import generic_filter
class lowpass_filter():
    def gaussian_filter(self,img):
        blurred_img = cv2.GaussianBlur(img, (3, 3), 0)
        return blurred_img

    def counterharmonic_mean_filter(self,x,Q):
        if np.sum(x ** Q) == 0:
            result =0
        else:
            result = np.sum(x ** (Q + 1)) / (np.sum(x ** Q))


        return result
    def contraharmonic_filter(self,img,Q):
        return generic_filter(img, lambda x: self.counterharmonic_mean_filter(x,Q), size=(3, 3, 3))




