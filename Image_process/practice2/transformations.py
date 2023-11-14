import cv2
import numpy as np

class transformations:

  def linear_trasformation_shift(self,img,tx,ty):
    M = np.float32([[1, 0, tx], [0, 1, ty]])
    translated_img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
    return translated_img

  def rotate_trasformation(self,img,angle,scale):
    height=img.shape[1]
    width=img.shape[0]
    center=[height/2,width/2]
    M = cv2.getRotationMatrix2D(center,angle,scale)
    rotated_image = cv2.warpAffine(img,M, (width, height))
    return rotated_image

  def scale_transformation(self,img,scale_percent):
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)

    scaled_img = cv2.resize(img, dim, interpolation=cv2.INTER_LINEAR)
    return scaled_img
  def projection_mapping(self,img,pts1,pts2):
    M = cv2.getPerspectiveTransform(pts1, pts2)
    perspective_img = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]))
    return perspective_img

  def affine_transformation(self,img,pts1,pts2):
    M = cv2.getAffineTransform(pts1, pts2)
    affine_img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
    return affine_img

  def Polynomial_Mapping(selfimg,img,coefficient,constant):
    height, width = img.shape[:2]
    map_x = np.zeros((height, width), np.float32)
    map_y = np.zeros((height, width), np.float32)
    for y in range(height):
      for x in range(width):
        map_x[y, x] = constant + coefficient*x + coefficient*y
        map_y[y, x] = constant + coefficient*x + coefficient*y
    remapped_img = cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR)
    return remapped_img

