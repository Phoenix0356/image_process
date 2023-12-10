import cv2
import numpy as np

import morphology as m

KERNEL = np.ones((3, 3), np.uint8)


morphology=m.morphology()

if __name__ == "__main__":
    # img = cv2.imread("2.jpg")
    # img_gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #
    # erode_img=morphology.erode_operation(img_gray,KERNEL)
    # dilate_img=morphology.dilate_operation(img_gray,KERNEL)
    #
    #
    # cv2.imshow("erode",erode_img)
    # cv2.imshow("dilate",dilate_img)
    # cv2.imshow("origin",img_gray)


    # 读取图像
    img = cv2.imread('2.jpg', cv2.IMREAD_GRAYSCALE)

    # 将图像二值化
    _, binary_img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    #close_operation
    close_img = morphology.dilate_operation(img,KERNEL,5)
    close_img = morphology.erode_operation(close_img, KERNEL, 2)
    #open_operation
    open_img = morphology.erode_operation(close_img,KERNEL,2)
    open_img = morphology.dilate_operation(open_img,KERNEL,5)

    cv2.imshow("split_img",open_img)
    cv2.imshow("original",img)

    cv2.waitKey(0)
