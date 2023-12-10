import cv2
import numpy as np
from scipy.signal import wiener

class nonlinear_filter():
    def median_filter(self,img,ksize):
        result = cv2.medianBlur(img, ksize)
        return result
    def weighted_median_filter(self,img,weights):
        rows, cols,channels = img.shape
        wsize = weights.shape[0]
        pad_size = wsize // 2

        padded_img = np.pad(img, pad_size, mode='edge')

        result = np.zeros_like(img)
        for i in range(rows):
            for j in range(cols):
                for k in range(channels):
                    window = padded_img[i:i + wsize, j:j + wsize, k]
                    weighted_window = np.repeat(window.flatten(), weights.flatten())
                    sorted_window = np.sort(weighted_window)
                    result[i, j, k] = sorted_window[len(sorted_window) // 2]
                    print("weighted_median_filter:",i," ", j," ",k)

        return result

    def rank_filter(self,img, ksize, rank):
        rows, cols,channels=img.shape
        pad_size = ksize // 2
        result = np.zeros_like(img, dtype=np.uint8)
        padded_image = np.pad(img, pad_size, mode='edge')

        for i in range(rows):
            for j in range(cols):
                for k in range(channels):
                    neighbors = padded_image[i:i + ksize, j:j + ksize, k]
                    sorted_neighbors = np.sort(neighbors.flatten())
                    result[i, j, k] = sorted_neighbors[rank]
                    print("rank_filter:",i," ",j," ",k)
        return result

    import cv2
    import numpy as np

    def wiener_filter(self,img, ksize, noise_variance):
        kernel = np.ones((ksize, ksize), dtype=np.float32) / (ksize ** 2)
        local_mean = cv2.filter2D(img, -1, kernel)
        local_var = cv2.filter2D(img ** 2, -1, kernel) - local_mean ** 2
        result = img + (noise_variance - local_var)
        result = np.clip(result, 0, 255).astype(np.uint8)
        return result

    def adaptive_median_filter(self,img, limit):
        rows, cols, channels = img.shape
        result = np.zeros_like(img)
        for k in range(channels):
            for i in range(rows):
                for j in range(cols):
                    size = 3
                    while size <= limit:
                        # 获取邻域像素值
                        neighbors = img[max(0, i - size // 2):min(rows, i + size // 2 + 1),
                                    max(0, j - size // 2):min(cols, j + size // 2 + 1), k].flatten()
                        median_value = np.median(neighbors)
                        min_value = np.min(neighbors)
                        max_value = np.max(neighbors)

                        if min_value < median_value < max_value:
                            if min_value < img[i, j, k] < max_value:
                                result[i, j, k] = img[i, j, k]
                            else:
                                result[i, j, k] = median_value
                            break
                        else:
                            size += 2
                        print("adaptive_filter:",k," ",i," ",j)
        return result






