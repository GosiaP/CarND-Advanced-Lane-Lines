import cv2
import numpy as np
import lane_find_utils as utl
import matplotlib.image as mpimg
import os
from camera_calib import CameraCalibration
from image_processor import BinaryImageCreator
from moviepy.editor import VideoFileClip
import glob

class RegionOfInterest:

    def __init__(self, image_size,  width2_p=0.125, height1_p=0.07, height2_p=0.375):
        self.image_size = image_size

        h = self.image_size[0]
        w = self.image_size[1]

        # points of region in clockwise orientation
        self.pts = np.float32([
            [w * 0.5 * (1 - width2_p), h * (1 - height2_p)],
            [w * 0.5 * (1 + width2_p), h * (1 - height2_p)],
            [w , h * (1 - height1_p)],
            [0 , h * (1 - height1_p)]])

        offsetx = 150
        offsety = 0

        self.dst = np.float32([
            [offsetx, offsety],
            [image_size[1] - offsetx, offsety],
            [image_size[1] - offsetx, image_size[0] - offsety],
            [offsetx, image_size[0] - offsety]])

        self.M = cv2.getPerspectiveTransform(self.pts, self.dst)
        self.M_inv = cv2.getPerspectiveTransform(self.dst, self.pts)

    def draw_into_image(self, image):
        cv2.polylines(image, [self.pts.astype(np.int32)], isClosed=True, color=(255,20,147), thickness=3)


# Functionality to detect a lane lanes
class LaneDetector:
    # Constructs the processor with given image/frame size
    def __init__(self, image_size):
        self.camera_cal = CameraCalibration()
        self.binimg_creator = BinaryImageCreator()
        self.region = RegionOfInterest(image_size)
        self.image_size = image_size

        self.lane_line_history = []
        self.lane_lines = None
        self.y_axis = np.linspace(0, image_size[0] - 1, num=image_size[0]).astype(np.int32)

    # Retrieves x values from lane line polynomials
    def get_lane_lines_x(self, lane_lines):
        if lane_lines is not None:
            lf = lane_lines[0]
            rf = lane_lines[1]
            y = self.y_axis
            left_x = lf[0] * y ** 2 + lf[1] * y + lf[2]
            right_x = rf[0] * y ** 2 + rf[1] * y + rf[2]
            return left_x, right_x
        return None

    # Adds low-pass filter to final lane lines
    def filter_lanes(self, img, lane_lines, window_size=10):
        # Append x values
        self.lane_line_history.append(self.get_lane_lines_x(lane_lines))

        # Enforce history size
        if len(self.lane_line_history) > window_size:
            self.lane_line_history.pop(0)

        # Calculate mean based on history
        x_mean_left = np.zeros(img.shape[0])
        x_mean_right = np.zeros(img.shape[0])
        x_mean_count = 0
        for lane_lines in self.lane_line_history:
            if lane_lines is not None:
                x_mean_left += lane_lines[0]
                x_mean_right += lane_lines[1]
                x_mean_count += 1

        if x_mean_count > 0:
            # Match new poly to mean
            x_mean_left = (x_mean_left / x_mean_count).astype(np.int32)
            x_mean_right = (x_mean_right / x_mean_count).astype(np.int32)
            lane_lines = utl.match_poly(img, (x_mean_left, self.y_axis, x_mean_right, self.y_axis))
            if utl.are_lines_valid(img, lane_lines):
                # Return mean
                return lane_lines
            else:
                # Corrupted history, restart once again
                self.lane_line_history.clear()
                return None
        else:
            # Nothing usable in history
            return None


    # Image processing pipeline
    def process(self, img):
        # Undistort
        img = self.camera_cal.undistort(img)
        # Color + gradient treshold
        top_down = self.binimg_creator.combi_treshold(img)
        # Perspective mapping
        top_down = cv2.warpPerspective(top_down, self.region.M, (top_down.shape[1], top_down.shape[0]), flags=cv2.INTER_LINEAR)

        # Lane point identification
        if self.lane_lines is not None:
            # Incremental approach
            lane_points = utl.find_lane_points_inc(top_down, self.lane_lines)
        else:
            # From scratch
            lane_points = utl.find_lane_points(top_down)

        # Calculate lane lane lines as polynomials
        self.lane_lines = utl.match_poly(top_down, lane_points)

        # Check if lanes are meaningful and push them to history filter
        if utl.are_lines_valid(top_down, self.lane_lines):
            self.lane_lines = self.filter_lanes(top_down, self.lane_lines)
        else:
            self.lane_lines = self.filter_lanes(top_down, None)

        if self.lane_lines is not None:
            # Calculate stats from lane poly lines
            curve_rad = utl.calc_curve_radius(self.y_axis, self.lane_lines)
            car_offset = utl.calc_car_offset(self.image_size, self.lane_lines)

            # Draw all results into image
            img = utl.draw_lane_area(img, self.lane_lines, self.region.M_inv)
            img = utl.draw_info(img, curve_rad, car_offset)
        return img



if __name__ == '__main__':

    video_file = 'project_video.mp4'
    #video_file = 'challenge_video.mp4'
    video_path = os.path.join('output_videos', video_file)
    clip = VideoFileClip(video_file)
    #clip = VideoFileClip(video_file).subclip(0,3)
    processor = LaneDetector((clip.h, clip.w, 3))
    new_clip = clip.fl_image(processor.process)
    new_clip.write_videofile(video_path, audio=False)

    """
    # Single image processing
    print("Testing lane tracker ....")
    image = mpimg.imread('test_images/straight_lines1.jpg')

    processor = LaneTracker(image.shape)
    res_img = processor.process(image)
    BinaryImageCreator.plot_images(image, res_img)
    print("Testing lane tracker")
    """


    




