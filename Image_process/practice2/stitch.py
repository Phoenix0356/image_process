import cv2
import numpy as np
class stitch():
    def crop_with_overlaping(self,img,y):
        crop_img_1=img[:,:y]
        crop_img_2=img[:,y-100:]
        crop_img_1=crop_img_1.astype(np.uint8)
        crop_img_2=crop_img_2.astype(np.uint8)


        return crop_img_1,crop_img_2
    def stitch(self,img1,img2):
        stitcher = cv2.Stitcher_create()
        status, panorama = stitcher.stitch([img1, img2])
        return panorama


