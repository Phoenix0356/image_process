import cv2 as cv
import numpy as np
import Histogram as h
import Profile as p
import Projection as pj
import matplotlib.pyplot as plt

# Draw a plot in a given image drawing context
# @param[in, out] image image drawing context
# @param[in] data_array data to draw
# @param[in] color color to use when drawing
# @param[in] max_val scale factor for the histogram values (default is 1)
def DrawGraph(image, data_array, color, max_val = 1.0):
  image_w = image.shape[1]
  image_h = image.shape[0]
  data_size = data_array.shape[0]

  step = image_w / data_size
  x = step * 0.5
  cv.line(image,
          (0, image_h - 1 - int((image_h - 1) * data_array[0] / max_val)),
          (int(x), image_h - 1 - int((image_h - 1) * data_array[0] / max_val)),
          color, thickness = 1)

  for i in range(1, data_size):
    cv.line(image,
            (int(x), image_h - 1 - int((image_h - 1) * data_array[i - 1] / max_val)),
            (int(x + step), image_h - 1 - int((image_h - 1) * data_array[i] / max_val)),
            color, thickness = 1)
    x += step

  cv.line(image,
          (int(x), image_h - 1 - int((image_h - 1) * data_array[data_size - 1] / max_val)),
          (image_w - 1, image_h - 1 - int((image_h - 1) * data_array[data_size - 1] / max_val)),
          color, thickness = 1)

# Draw a histogram in a given image drawing context
# @param[in, out] image image drawing context
# @param[in] data_array data to draw
# @param[in] color color to use when drawing
# @param[in] max_val scale factor for the histogram values (default is 1)
def DrawHist(image, data_array, color, max_val = 1.0):
  image_w = image.shape[1]
  image_h = image.shape[0]
  data_size = data_array.shape[0]

  step = image_w / data_size
  x = 0
  for i in range(0, data_size):
    cv.rectangle(image,
                 (int(x), image_h - 1 - int((image_h - 1) * data_array[i] / max_val)),
                 (int(x + step) - 1, image_h - 1),
                 color, thickness = -1)
    x += step

# Export a matrix into a file
# @param[out] fn file name
# @param[in] mat matrix to export
# @param[in] delimiter delimiter between columns
def ExportText(fn, mat, delimiter = '\t'):
  if mat.ndim > 2:
    mat_out = mat.reshape(mat.shape[0], -1)
  else:
    mat_out = mat
  if mat.dtype == np.uint8:
    format = "%i"
  else:
    format = "%f"
  np.savetxt(fn, mat_out, fmt = format, delimiter = delimiter)


def get_img_hist(img):
    hist = cv.calcHist([img], [0], None, [256], [0, 256])
    cum_hist = np.cumsum(hist) / img.shape[0] / img.shape[1]
    hist_img = np.full((256, 512, 3), 255, dtype=np.uint8)
    DrawHist(hist_img, hist, (127, 127, 127), hist.max())
    DrawGraph(hist_img, cum_hist, (0, 0, 0), 1)
    return hist_img

# # #
# Main function
# # #

#if __name__ == "__main__":
#     #Histogram
#     img = cv.imread("bb8.jpg", cv.IMREAD_COLOR)
#     if not isinstance(img, np.ndarray) or img.data is None:
#         print("Error reading file \"{}\"".format(fn))
#         exit()
#
#     img_bw = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
#
#     l=h.Lut()
#
#
#     hist_origin=get_img_hist(img_bw)
#
#     img_processed=l.linear_equalization(img_bw)
#
#     hist_processed=get_img_hist(img_processed)
#
#
#     img_processed_2 = l.nonlinear_dynamic_stretching(img_bw)
#
#     hist_processed_2 = get_img_hist(img_processed_2)
#
#     fig, axs = plt.subplots(3, 2)
#     axs[0, 0].imshow(img_bw,cmap='gray')
#     axs[0, 0].set_title('Original Image')
#     axs[0, 0].axis('off')
#
#     axs[0, 1].imshow(hist_origin, cmap='gray')
#     axs[0, 1].set_title('Histogram of Original Image')
#     axs[0, 1].axis('off')
#
#     axs[1, 0].imshow(img_processed, cmap='gray')
#     axs[1, 0].set_title('Processed Image')
#     axs[1, 0].axis('off')
#
#     axs[1, 1].imshow(hist_processed, cmap='gray')
#     axs[1, 1].set_title('Histogram of Processed Image')
#     axs[1, 1].axis('off')
#
#     axs[2, 0].imshow(img_processed_2, cmap='gray')
#     axs[2, 0].set_title('Processed Image2')
#     axs[2, 0].axis('off')
#
#     axs[2, 1].imshow(hist_processed_2, cmap='gray')
#     axs[2, 1].set_title('Histogram of Processed Image2')
#     axs[2, 1].axis('off')
#     plt.show()
# ----------------------------------------------------------- #
#   # Profile
#     img_barcode = cv.imread('bar_code.jpg', cv.IMREAD_COLOR)
#
#     profile=p.profile()
#     img_profile=profile.profile_barcode(img_barcode)
#     fig, axs = plt.subplots(1,2)
#     axs[0].imshow(img_barcode,cmap='gray')
#     axs[0].set_title('barcode')
#     axs[0].axis('off')
#
#     axs[1].plot(img_profile,color="black")
#     axs[1].set_title('profile of barcode')
#     axs[1].axis('off')
#     plt.show()
# ----------------------------------------------------------- #
#     #Projection
#     img_bw = cv.imread("TIE.jpg", cv.IMREAD_GRAYSCALE)
#     img_height = img_bw.shape[0]
#     img_width = img_bw.shape[1]
#
#     pj=pj.projection()
#     vertical_projection, horizontal_projection=pj.img_project(img_bw)
#
#     vertical_contours = pj.find_obj(vertical_projection,is_height=True
#                                   ,height=img_height,width=img_width)
#     horizontal_contours = pj.find_obj(horizontal_projection, is_height=False
#                                     , height=img_height, width=img_width)
#
#     top_left=[vertical_contours[0],horizontal_contours[0]]
#     bottom_right=[vertical_contours[1],horizontal_contours[1]]
#
#     img_bw=pj.draw_target(img_bw,top_left,bottom_right)
#     pj.show_projection(vertical_projection,vertical_contours,horizontal_projection,horizontal_contours)
#     cv.imshow("target_find",img_bw)
#     cv.waitKey(0)
#     cv.destroyAllWindows()





