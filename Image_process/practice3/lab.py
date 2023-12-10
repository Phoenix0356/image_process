import noises_types as nt
import lowpass_filter as lf
import nonlinear_filter as nf
import highpass_filter as hf

import cv2
import numpy as np

IMPULSE_INTENSITY=50000
QUANTIZATION_INTENSITY=100

Q_ARRAY=[-1,0,1]
WEIGHT_MATRIX = np.array([[1, 2, 1],
                    [2, 4, 2],
                    [1, 2, 1]])


img = cv2.imread('bb8.jpg')
noise_type=nt.noises_types()
low_pass_filter=lf.lowpass_filter()
nonlinear_filter=nf.nonlinear_filter()
high_pass_filter=hf.hightpass_filter()

if __name__ == "__main__":



    # gaussian_img=noise_type.guassian_noise(img)
    # impulse_img=noise_type.impulse_noise(img,IMPULSE_INTENSITY)
    # multiple_img=noise_type.multiple_noise(img)
    # quantization_img=noise_type.qantization_noise(img,QUANTIZATION_INTENSITY)
    # cv2.imshow("original_img", img)
    # cv2.imshow("gaussian_img",gaussian_img)
    # cv2.imshow("impulse_img",impulse_img)
    #cv2.imshow("multiple_img",multiple_img)
    # cv2.imshow("quantization_img",quantization_img)

    # ----------------------------------------------------------- #

    # gaussian_img = noise_type.guassian_noise(img)
    # gaussian_filter_img=low_pass_filter.gaussian_filter(gaussian_img)
    # for i in range(3):
    #     print("contraharmonic_filter:", i)
    #     contraharmonic_filter_img=low_pass_filter.contraharmonic_filter(img,Q_ARRAY[i])
    #     cv2.imshow("contraharmonic_filter_img{}".format(i), contraharmonic_filter_img)
    #
    # cv2.imshow("gaussian_filter_img",gaussian_filter_img)


    # ----------------------------------------------------------- #

    # median_img=nonlinear_filter.median_filter(img,7)
    # weighted_median_img=nonlinear_filter.weighted_median_filter(img,WEIGHT_MATRIX)
    # rank_img=nonlinear_filter.rank_filter(img,3,2)
    # wiener_img=nonlinear_filter.wiener_filter(img,5,2)
    # adaptive_img=nonlinear_filter.adaptive_median_filter(img,10)

    # cv2.imshow("median_filter_img", median_img)
    # cv2.imshow("weighted_median_filter_img", weighted_median_img)
    # cv2.imshow("rank_filter_img", rank_img)
    # cv2.imshow("wiener_filter_img",wiener_img)
    # cv2.imshow("adaptive_filter_img",adaptive_img)

    # cv2.imwrite("C:/Users/Phoenix/Desktop/IP/practice3/picture/median_filter_img.jpg", median_img)
    # cv2.imwrite("C:/Users/Phoenix/Desktop/IP/practice3/picture/weighted_median_filter_img.jpg", weighted_median_img)
    # cv2.imwrite("C:/Users/Phoenix/Desktop/IP/practice3/picture/rank_filter_img.jpg", rank_img)
    # cv2.imwrite("C:/Users/Phoenix/Desktop/IP/practice3/picture/wiener_filter_img.jpg", wiener_img)
    # cv2.imwrite("C:/Users/Phoenix/Desktop/IP/practice3/picture/adaptive_filter_img.jpg", adaptive_img)

    # ----------------------------------------------------------- #

    # robert_img=high_pass_filter.roborts(img)
    # prewitt_img=high_pass_filter.prewitts(img)
    # sobal_img=high_pass_filter.sobel(img)
    # laplace_img=high_pass_filter.laplace(img)
    # canny_img=high_pass_filter.canny(img)
    #
    # cv2.imshow("roberts",robert_img)
    # cv2.imshow("prewitt",prewitt_img)
    # cv2.imshow("sobal",prewitt_img)
    # cv2.imshow("laplace",laplace_img)
    # cv2.imshow("canny",canny_img)


    cv2.waitKey(0)
