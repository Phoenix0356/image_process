import numpy as np
import cv2
class noises_types():
    def guassian_noise(self,img):
        img = img.astype(float)
        mean = 1
        var = 5
        sigma = var ** 0.5
        gaussian = np.random.normal(mean, sigma, img.shape)
        gaussian = gaussian.reshape(img.shape).astype(float)
        noisy_image = cv2.addWeighted(img, 0.5, gaussian, 0.3, 0.1)
        noisy_image=noisy_image.astype(np.uint8)
        return noisy_image



    def impulse_noise(self,img,intensity):
        noisy_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        for i in range(intensity):

            row = np.random.randint(0, noisy_image.shape[0])
            col = np.random.randint(0, noisy_image.shape[1])

            if np.random.rand() < 0.5:
                noisy_image[row, col] = 0
            else:
                noisy_image[row, col] = 255

        return noisy_image

    def multiple_noise(self,img):
        img = img / 255
        noise = np.random.uniform(0.75, 1.25, img.shape)

        noisy_img = cv2.multiply(img, noise)  # 将噪声与图像相乘
        noisy_img = (noisy_img * 255).astype(np.uint8)
        return noisy_img

    def qantization_noise(self,img, intensity):
        noisy_image = img.copy()
        noise = np.random.randint(-intensity, intensity + 1, noisy_image.shape)
        noisy_image = np.clip(noisy_image + noise, 0, 255).astype(np.uint8)
        return noisy_image








