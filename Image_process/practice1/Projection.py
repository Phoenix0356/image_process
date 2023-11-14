import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# 读取图像
class projection():
    def img_project(self,img):
        vertical_projection = np.sum(img, axis=0)
        horizontal_projection = np.sum(img, axis=1)
        return vertical_projection,horizontal_projection

    def find_obj(self,projection, is_height, height, width):
        boundaries = []
        length = height if is_height else width
        find_object = False
        for i in range(len(projection)):
            if not find_object and projection[i] - length * 254 < 0:
                boundaries.append(i)
                find_object = True
            elif find_object and projection[i] - length * 254 >= 0:
                boundaries.append(i - 1)
                find_object = False
        if find_object:
            boundaries.append(len(projection) - 1)
        return boundaries

    def draw_target(self,img,top_left,bottom_right):
        cv.rectangle(img, top_left, bottom_right, (0,0,0), 2)
        return img

    def show_projection(self,vertical_projection,vertical_boundaries,horizontal_projection,horizontal_boundaries):
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(vertical_projection)
        plt.title('Vertical Projection')

        for boundary in vertical_boundaries:
            plt.axvline(boundary, color='green', linewidth=1)

        plt.subplot(1, 2, 2)
        plt.plot(horizontal_projection)
        plt.title('Horizontal Projection')
        for boundary in horizontal_boundaries:
            plt.axvline(boundary, color='green', linewidth=1)
        plt.show()