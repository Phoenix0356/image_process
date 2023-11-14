import transformations as tf
import distortion as d
import stitch as s
import cv2
import numpy as np
def bilinear_inverse(p, vertices, numiter=4):
  p = np.asarray(p)
  v = np.asarray(vertices)
  sh = p.shape[1:]
  if v.ndim == 2:
    v = np.expand_dims(v, axis=tuple(range(2, 2 + len(sh))))

  # Start in the center
  s = .5 * np.ones((2,) + sh)
  s0, s1 = s
  for k in range(numiter):
    # Residual
    r = v[0] * (1 - s0) * (1 - s1) + v[1] * s0 * (1 - s1) + v[2] * s0 * s1 + v[3] * (1 - s0) * s1 - p

    # Jacobian
    J11 = -v[0, 0] * (1 - s1) + v[1, 0] * (1 - s1) + v[2, 0] * s1 - v[3, 0] * s1
    J21 = -v[0, 1] * (1 - s1) + v[1, 1] * (1 - s1) + v[2, 1] * s1 - v[3, 1] * s1
    J12 = -v[0, 0] * (1 - s0) - v[1, 0] * s0 + v[2, 0] * s0 + v[3, 0] * (1 - s0)
    J22 = -v[0, 1] * (1 - s0) - v[1, 1] * s0 + v[2, 1] * s0 + v[3, 1] * (1 - s0)

    inv_detJ = 1. / (J11 * J22 - J12 * J21)

    s0 -= inv_detJ * (J22 * r[0] - J12 * r[1])
    s1 -= inv_detJ * (-J21 * r[0] + J11 * r[1])

  return s

def invert_map(xmap, ymap, diagnostics=False):

  # Generate quadrilaterals from mapped grid points.
  quads = np.array([[ymap[:-1, :-1], xmap[:-1, :-1]],
            [ymap[1:, :-1], xmap[1:, :-1]],
            [ymap[1:, 1:], xmap[1:, 1:]],
            [ymap[:-1, 1:], xmap[:-1, 1:]]])

  # Range of indices possibly within each quadrilateral
  x0 = np.floor(quads[:, 1, ...].min(axis=0)).astype(int)
  x1 = np.ceil(quads[:, 1, ...].max(axis=0)).astype(int)
  y0 = np.floor(quads[:, 0, ...].min(axis=0)).astype(int)
  y1 = np.ceil(quads[:, 0, ...].max(axis=0)).astype(int)

  # Quad indices
  i0, j0 = np.indices(x0.shape)

  # Offset of destination map
  x0_offset = x0.min()
  y0_offset = y0.min()

  # Index range in x and y (per quad)
  xN = x1 - x0 + 1
  yN = y1 - y0 + 1

  # Shape of destination array
  sh_dest = (1 + x1.max() - x0_offset, 1 + y1.max() - y0_offset)

  # Coordinates of destination array
  yy_dest, xx_dest = np.indices(sh_dest)

  xmap1 = np.zeros(sh_dest)
  ymap1 = np.zeros(sh_dest)
  TN = np.zeros(sh_dest, dtype=int)

  # Smallish number to avoid missing point lying on edges
  epsilon = .01

  # Loop through indices possibly within quads
  for ix in range(xN.max()):
    for iy in range(yN.max()):
      # Work only with quads whose bounding box contain indices
      valid = (xN > ix) * (yN > iy)

      # Local points to check
      p = np.array([y0[valid] + ix, x0[valid] + iy])

      # Map the position of the point in the quad
      s = bilinear_inverse(p, quads[:, :, valid])

      # s out of unit square means p out of quad
      # Keep some epsilon around to avoid missing edges
      in_quad = np.all((s > -epsilon) * (s < (1 + epsilon)), axis=0)

      # Add found indices
      ii = p[0, in_quad] - y0_offset
      jj = p[1, in_quad] - x0_offset

      ymap1[ii, jj] += i0[valid][in_quad] + s[0][in_quad]
      xmap1[ii, jj] += j0[valid][in_quad] + s[1][in_quad]

      # Increment count
      TN[ii, jj] += 1

  ymap1 /= TN + (TN == 0)
  xmap1 /= TN + (TN == 0)

  if diagnostics:
    diag = {'x_offset': x0_offset,
        'y_offset': y0_offset,
        'mask': TN > 0}
    return xmap1, ymap1, diag
  else:
    return xmap1, ymap1
#shift
X_SHIFT=200
Y_SHIFT=100
#rotate
ANGLE=60
SCALE=0.8
#scale
SCALE=50
#affine
AFFINE1 = np.float32([[0, 0], [326, 300], [200,500]])
AFFINE2 = np.float32([[100, 100], [250, 150], [200, 300]])
#projection
PROJECTION1= np.float32([[0,0], [652,0], [0,778], [652, 778]])
PROJECTION2 = np.float32([[652, 778], [652,0], [0,778], [0, 0]])
#polynomial_mapping
COEFFICIENT=0.5
CONSTANT=1
#pincushion_distortion
K=5

img=cv2.imread('bb8.jpg')

# # #
# Main function
# # #
if __name__ == "__main__":

    # transformation= tf.transformations()
    #
    # shift_img=transformation.linear_trasformation_shift(img,X_SHIFT,Y_SHIFT)
    # rotate_img=transformation.rotate_trasformation(img,ANGLE,SCALE)
    # scaled_img=transformation.scale_transformation(img,SCALE)
    # affine_img=transformation.affine_transformation(img,AFFINE1,AFFINE2)
    # projection_img = transformation.projection_mapping(img, PROJECTION1, PROJECTION2)
    # polynomial_mapping_img=transformation.Polynomial_Mapping(img,COEFFICIENT,CONSTANT)

    # cv2.imshow("origin", img)
    # cv2.imshow("shift",shift_img)
    # cv2.imshow("rotate",rotate_img)
    #cv2.imshow("scale",scaled_img)
    # cv2.imshow("projection",projection_img)
    # cv2.imshow("affine",affine_img)
    # cv2.imshow("polynomial_img",polynomial_mapping_img)

# ----------------------------------------------------------- #

    # distortion=d.distortion()
    # distorted_img=distortion.pincushion_distortion(img)
    # correct_img=distortion.correct(distorted_img)
    # cv2.imshow("origin", img)
    # cv2.imshow("distortion", distorted_img)
    # cv2.imshow("correct", correct_img)

    # cv2.imwrite("C:/Users/Phoenix/Desktop/IP/practice2/pictures/distortion.jpg", distorted_img)
    # cv2.imwrite("C:/Users/Phoenix/Desktop/IP/practice2/pictures/correct.jpg", correct_img)


# ----------------------------------------------------------- #
#     stitch=s.stitch()
#     crop_img1,crop_img2=stitch.crop_with_overlaping(img,439)
#     stitch_img=stitch.stitch(crop_img1,crop_img2)
#     cv2.imshow("origin", img)
#     cv2.imshow("crop1", crop_img1)
#     cv2.imshow("crop2", crop_img2)
#     cv2.imshow("stitch", stitch_img)
    cv2.waitKey(0)