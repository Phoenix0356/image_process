import cv2
import numpy as np

class distortion():
    def pincushion_distortion(self,img):
        height, width = img.shape[:2]


        map_x = np.zeros((height, width), np.float32)
        map_y = np.zeros((height, width), np.float32)

        for y in range(height):
            for x in range(width):
                r = np.sqrt((x - width / 2) ** 2 + (y - height / 2) ** 2)
                theta = np.arctan2(y - height / 2, x - width / 2)
                r = r * (0.8 + 0.001 * r)
                map_x[y, x] = r * np.cos(theta) + width / 2
                map_y[y, x] = r * np.sin(theta) + height / 2

        distorted_img = cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR)
        return distorted_img
    def correct(self,distorted_img):
        height, width = distorted_img.shape[:2]

        map_x = np.zeros((height, width), np.float32)
        map_y = np.zeros((height, width), np.float32)
        for y in range(height):
            for x in range(width):
                r = np.sqrt((x - width / 2) ** 2 + (y - height / 2) ** 2)
                theta = np.arctan2(y - height / 2, x - width / 2)
                r = r / (0.8 + 0.001 * r)
                map_x[y, x] = r * np.cos(theta) + width / 2
                map_y[y, x] = r * np.sin(theta) + height / 2

        corrected_img = cv2.remap(distorted_img, map_x, map_y, cv2.INTER_LINEAR)
        return corrected_img