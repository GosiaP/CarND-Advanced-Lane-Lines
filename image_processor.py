import cv2
import numpy as np
from camera_calib import CameraCalibration
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import glob
# Change to appropriate color map for gray scale images
plt.rcParams['image.cmap'] = 'gray'


# Creates binary images that are result of color and gradient threshold
class BinaryImageCreator:
    def __init__(self):
        self.kernel_size = 7 # Sobel kernel size

    # Uses Sobel x or y,
    # then takes an absolute value and applies a threshold.
    def abs_sobel_thresh(self, sobel, thresh_min, thresh_max):
        abs_sobel = np.absolute(sobel)
        scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))

        binary_img = np.zeros_like(scaled_sobel)
        binary_img[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1
        return binary_img


    # Uses Sobel x and y,
    # then computes the magnitude of the gradient
    # and applies a threshold
    def mag_thresh(self, sobelx, sobely, thresh_min=0, thresh_max=255):
        magn = np.sqrt(sobelx ** 2 + sobely ** 2)
        scale_factor = np.max(magn) / 255
        magn = (magn / scale_factor).astype(np.uint8)

        binary_img = np.zeros_like(magn)
        binary_img[(magn >= thresh_min) & (magn <= thresh_max)] = 1
        return binary_img


    # Uses Sobel x and y,
    # then computes the direction of the gradient
    # and applies a threshold.
    def dir_threshold(self, sobelx, sobely, thresh_min=0, thresh_max=np.pi / 2):
        dir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))

        binary_img = np.zeros_like(dir)
        binary_img[(dir >= thresh_min) & (dir <= thresh_max)] = 1
        return binary_img

    def sobel_treshold(self, img):
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        sobelx = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, self.kernel_size)
        sobely = cv2.Sobel(gray_img, cv2.CV_64F, 0, 1, self.kernel_size)

        thresh_min = 20
        thresh_max = 255

        # apply each of the threshold functions
        gradx = self.abs_sobel_thresh(sobelx, thresh_min, thresh_max)
        grady = self.abs_sobel_thresh(sobely, thresh_min, thresh_max)
        mag_binary = self.mag_thresh(sobelx, sobely, thresh_min=30, thresh_max=255)
        dir_binary = self.dir_threshold(sobelx, sobely, thresh_min=.7, thresh_max=1.3)

        # combine thresholds
        binary = np.zeros_like(gradx)
        binary[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1
        return binary

    # Threshold a white color using HSL channels
    def color_treshold_yellow(self, h_chanel, l_chanel, s_chanel):
        # yellow color mask
        #lower = np.uint8([10, 0, 100])
        #upper = np.uint8([40, 255, 255])

        # threshold yellow color
        binary = np.zeros_like(h_chanel)
        binary[((h_chanel > 10) & (h_chanel <= 40) & (l_chanel > 0) & (l_chanel <= 255) & (s_chanel > 100) & (s_chanel <= 255))] = 1
        return binary

    # Threshold a yellow color using HSL channels
    def color_treshold_white(self, h_chanel, l_chanel, s_chanel):
        # white color mask
        #lower = np.uint8([0, 200, 0])
        #upper = np.uint8([255, 255, 255])

        # threshold white color
        binary = np.zeros_like(h_chanel)
        binary[((h_chanel > 0) & (h_chanel <= 255) & (l_chanel > 200) & (l_chanel <= 255) & (s_chanel > 0) & (s_chanel <= 255))] = 1
        return binary

    # Threshold a color using S channel from HSL color image
    def color_treshold(self, img):
        # convert to HLS color space
        hls_img = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)

        # separate  channels
        s_chanel = hls_img[:, :, 2]

        # threshold white color
        binary = np.zeros_like(s_chanel)
        binary[( (s_chanel > 170) & (s_chanel <= 255))] = 1
        return binary

    # Combines gradient (Sobel) and color thresholds
    def combi_treshold(self, img):
        sobel_img = self.sobel_treshold(img)

        # convert to HLS color space and separate every channel
        hls_img = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        h_chanel = hls_img[:, :, 0]
        l_chanel = hls_img[:, :, 1]
        s_chanel = hls_img[:, :, 2]

        img_yellow = self.color_treshold_yellow(h_chanel, l_chanel, s_chanel);
        img_white = self.color_treshold_white(h_chanel, l_chanel, s_chanel);
        #img_color = self.color_treshold(img)

        binary = np.zeros_like(sobel_img)
        binary[(sobel_img == 1) | ((img_yellow == 1) | (img_white == 1))] = 1
        #binary[(sobel_img == 1) | (img_color == 1)] = 1

        return binary

    @staticmethod
    def plot_images(orig_img, trans_img, titel_1 = 'Original Image', title_2 = 'Processed image'):
        f = plt.figure(figsize=(15, 7))
        plt.subplot(1, 2, 1)
        f.tight_layout()
        plt.title(titel_1, fontsize=30)
        plt.imshow(orig_img)
        plt.subplot(1, 2, 2)
        plt.title(title_2, fontsize=30)
        plt.imshow(trans_img)
        return

    @staticmethod
    def plot_image_list(images, ncols):
        n = images.shape[0]
        if n <= 0:
            return

        nrows = int(n/ ncols)
        rest = (n % ncols)
        if rest > 0:
            nrows = nrows + 1

        f, ax = plt.subplots(nrows, ncols, figsize=(15, 10), frameon=False)
        plt.axis('off')
        f.subplots_adjust(hspace=0.02, wspace=0.05)

        i = 0
        for irow in range(nrows):
            for icol in range(ncols):
                if i >= len(images):
                    break;
                img = images[i]
                ax[irow, icol].axis('off')
                ax[irow, icol].imshow(img)
                i=i+1
        plt.show()
        return

# For testing purpose
if __name__ == '__main__':

    print("Testing of binary image ....")

    """
    image = mpimg.imread('test_images/test2.jpg')
    cam_calib = CameraCalibration()
    rr = cam_calib.undistort(image)
    BinaryImageCreator.plot_images(image, rr)
    """

    image_files = glob.glob('test_images/*.jpg')
    images_x = []
    for fn in image_files:
        images_x.append(mpimg.imread(fn))
    images = np.array(images_x)
    #BinaryImageCreator.plot_image_list(images, 3)

    # remove image distortion
    cam_calib = CameraCalibration()

    images_x.clear()
    for img in images:
        images_x.append(cam_calib.undistort(img))
    images = np.array(images_x)
    #BinaryImageCreator.plot_image_list(images, 3)

    # detect edges on binary image
    creator = BinaryImageCreator()

    images_x.clear()
    for img in images:
        images_x.append(creator.combi_treshold(img))
    images = np.array(images_x)
    BinaryImageCreator.plot_image_list(images, 3)

    print("Testing of binary image done")