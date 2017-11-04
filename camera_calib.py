import os
import pickle
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob


class CameraCalibration:

    def __init__(self):
        self.pickle_name = 'cam_calib.p'
        # this a folder where chessboard images are store and
        # where pickle file with camera parameters will be stored too
        self.folder_name = 'camera_cal'
        self.check_width=9
        self.check_height=6

        self.cam_calibration = self.__load_data__()
        if self.cam_calibration is  None:
            self.cam_calibration = self.__calibrate_camera__()
            self.__save_data__()

    def __load_data__(self):
        pickle_file_path =  os.path.join(self.folder_name, self.pickle_name)
        if os.path.isfile(pickle_file_path):
            with open(pickle_file_path, mode='rb') as f:
                return pickle.load(f)
        return None

    def __save_data__(self):
        pickle_file_path = os.path.join(self.folder_name, self.pickle_name)
        with open(pickle_file_path, mode='wb') as f:
            pickle.dump(self.cam_calibration, f)
        return

    def __calibrate_camera__(self):

        obj_corners = np.zeros((self.check_height * self.check_width, 3), np.float32)
        obj_corners[:, :2] = np.mgrid[0:self.check_width, 0:self.check_height].T.reshape(-1, 2)

        # Collect image chessboard corners (img_points) and associated normal chessboard corners (obj_points)
        img_points = []
        obj_points = []

        images = glob.glob('{}/calibration*.jpg'.format(self.folder_name))

        img_shape = [0,0]
        for image in images:
            img = mpimg.imread(image)
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) # images read with matplotlib is in RGB color space

            ret, img_corners = cv2.findChessboardCorners(gray, (self.check_width, self.check_height), None)
            if ret:
                img_points.append(img_corners)
                obj_points.append(obj_corners)
                img = cv2.drawChessboardCorners(img, (self.check_width, self.check_height), img_corners, ret)
                mpimg.imsave(image.replace('calibration', 'chessboard_corners'), img, format='jpg')

            img_shape = gray.shape[::-1]

        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, img_shape, None, None)

        return {'matrix': mtx, 'dist': dist}

    # Returns undistorted image
    def undistort(self, img):
        return cv2.undistort(
            img,
            self.cam_calibration['matrix'],
            self.cam_calibration['dist'],
            None,
            self.cam_calibration['matrix'])

    def plot_images(self, test_img_file_name):
        test_img_path = os.path.join(self.folder_name, test_img_file_name)

        if os.path.exists(test_img_path):
            test_img = mpimg.imread(test_img_path)
            undist = self.undistort(test_img)

            f = plt.figure(figsize=(15, 7))
            plt.subplot(1, 2, 1)
            f.tight_layout()
            plt.title('Original Image', fontsize=30)
            plt.imshow(test_img)
            plt.subplot(1, 2, 2)
            plt.title('Undistorted Image', fontsize=30)
            plt.imshow(undist)
        return



if __name__ == '__main__':
    print("Testing camera calibration...")
    cam_calib = CameraCalibration()
    cam_calib.plot_images('calibration2.jpg')
    print("Testing camera calibration done")
