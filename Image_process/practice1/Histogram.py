import cv2 as cv
import numpy as np
import os

class Lut():
    def linear_lut(self,img):
        lut = np.arange(256, dtype=np.uint8)
        return cv.LUT(img, lut)

    def linear_equalization(self,img):
        hist = cv.calcHist([img], [0], None, [256], [0, 256])
        cum_hist = np.cumsum(hist) / img.shape[0] / img.shape[1]
        lut = (255 * cum_hist).clip(0, 255).astype(np.uint8)
        return cv.LUT(img, lut)

    def nonlinear_dynamic_stretching(self,img):
        i_min, i_max = img.min(), img.max()
        alpha = 0.5
        lut = np.arange(256, dtype=np.float32)
        lut = np.power(((lut - i_min) / (i_max - i_min)).clip(0, 1), alpha)
        lut = (255 * lut).astype(np.uint8)
        return cv.LUT(img, lut)

    def uniform_transformation(self,img):
        i_min, i_max = img.min(), img.max()
        lut = (i_max - i_min) * cum_hist + i_min
        lut = lut.clip(0, 255).astype(np.uint8)
        return cv.LUT(img, lut)

    def exponential_transformation(self,img):
        alpha = 3
        lut = img.min() / 255.0 - 1.0 / alpha * np.log(1 - cum_hist)
        lut = (255 * lut).clip(0, 255).astype(np.uint8)
        return cv.LUT(img, lut)

    def rayleigh_transformation(self,img):
        alpha = 0.3
        lut = img.min() / 255 + np.sqrt(2 * alpha * alpha * np.log(1 / (1 - cum_hist)))
        lut = (255 * lut).clip(0, 255).astype(np.uint8)
        return cv.LUT(img, lut)

    def two_thirds_degree_transformation(self,img):
        lut = np.power(cum_hist, 2.0 / 3)
        lut = (255 * lut).clip(0, 255).astype(np.uint8)
        return cv.LUT(img, lut)

    def hyperbolic_transformation(self,img):
        alpha = 0.04
        lut = np.power(alpha, cum_hist)
        lut = (255 * lut).clip(0, 255).astype(np.uint8)
        return cv.LUT(img, lut)





