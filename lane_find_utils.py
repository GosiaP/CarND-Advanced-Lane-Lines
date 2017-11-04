import numpy as np
import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

# Define conversions in x and y from pixels space to meters
# assume the lane is about 30 meters long and 3.7 meters wide
X_METER_PER_PIXEL = 3.7 / 700
Y_METER_PER_PIXEL = 30 / 720


# Udacity code: Finds lane points using search window
def find_lane_points(img, nwindows=10, margin=100, minpix=50):
    # Take a histogram of the bottom half of the image
    histogram = np.sum(img[img.shape[0] // 2:, :], axis=0)

    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0] / 2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Set height of windows
    window_height = np.int(img.shape[0] / nwindows)

    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = img.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    ## Create an output image to draw on and  visualize the result
    out_img = np.dstack((img, img, img))*255

    leftdx_current = 0
    rightdx_current = 0

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = img.shape[0] - (window + 1) * window_height
        win_y_high = img.shape[0] - window * window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin

        # Draw the windows on the visualization image
        cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0, 255, 0), 2)
        cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0, 255, 0), 2)

        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            new_leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            leftdx_current = new_leftx_current - leftx_current
            leftx_current = new_leftx_current
        elif len(good_left_inds) == 0:
            leftx_current += leftdx_current
        if len(good_right_inds) > minpix:
            new_rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
            rightdx_current = new_rightx_current - rightx_current
            rightx_current = new_rightx_current
        elif len(good_right_inds) == 0:
            rightx_current += rightdx_current
    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    """
    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    ploty = np.linspace(0, img.shape[0] - 1, img.shape[0])
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
    plt.imshow(out_img)
    plt.plot(left_fitx, ploty, color='yellow')
    plt.plot(right_fitx, ploty, color='yellow')
    plt.xlim(0, 1280)
    plt.ylim(720, 0)
    """

    return leftx, lefty, rightx, righty


# Udacity code: Finds lane points based on previous lane points
def find_lane_points_inc(img, lane_lines, margin=100):
    # Assume you now have a new warped binary image
    # from the next frame of video (also called "binary_warped")
    # It's now much easier to find line pixels!
    nonzero = img.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    left_fit = lane_lines[0]
    right_fit = lane_lines[1]

    left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] + margin)))
    right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] + margin)))

    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    return leftx, lefty, rightx, righty


# Udacity code: Matches 2nd grade polynomial to lane points
def match_poly(img, lane_points):
    if len(lane_points[0]) < 2 or len(lane_points[2]) < 2:
        return None

    # Fit a second order polynomial to each list of lanes points (for left and right lanes)
    left_fit = np.polyfit(lane_points[1], lane_points[0], 2)
    right_fit = np.polyfit(lane_points[3], lane_points[2], 2)

    return left_fit, right_fit


# Udacity code: Calculates curve radius from lane lines
def calc_curve_radius(y_axis, lane_lines):
    left_fit = lane_lines[0]
    right_fit = lane_lines[1]

    # Generate x and y values for plotting
    left_fitx = left_fit[0]*y_axis**2 + left_fit[1]*y_axis + left_fit[2]
    right_fitx = right_fit[0]*y_axis**2 + right_fit[1]*y_axis + right_fit[2]

    # Define y-value where we want radius of curvature
    # I'll choose the maximum y-value, corresponding to the bottom of the image
    y_eval = np.max(y_axis)

    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(y_axis * Y_METER_PER_PIXEL, left_fitx * X_METER_PER_PIXEL, 2)
    right_fit_cr = np.polyfit(y_axis * Y_METER_PER_PIXEL, right_fitx * X_METER_PER_PIXEL, 2)

    # Calculate the new radius of curvature
    left_curverad = ((1 + (2 * left_fit_cr[0] * y_eval * Y_METER_PER_PIXEL + left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(2 * left_fit_cr[0])
    right_curverad = ((1 + (2 * right_fit_cr[0] * y_eval * Y_METER_PER_PIXEL + right_fit_cr[1]) ** 2) ** 1.5) / np.absolute(2 * right_fit_cr[0])

    # Now our radius of curvature is in meters
    return left_curverad, right_curverad


# Udacity code: Draws lane area into undistorted image
def draw_lane_area(img, lane_lines, M_inv):
    left_fit = lane_lines[0]
    right_fit = lane_lines[1]

    # Create an image to draw the lines on
    color_warp = np.zeros_like(img).astype(np.uint8)

    # Recast the x and y points into usable format for cv2.fillPoly()
    ploty = np.linspace(0, img.shape[0] - 1, num=img.shape[0])
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, M_inv, (img.shape[1], img.shape[0]))
    # Combine the result with the original image
    return cv2.addWeighted(img, 1, newwarp, 0.3, 0)


# Calculates car offset from lane center
def calc_car_offset(image_size, lane_lines):
    left_fit = lane_lines[0]
    right_fit = lane_lines[1]

    y = image_size[0]
    left_fitx = left_fit[0] * y ** 2 + left_fit[1] * y + left_fit[2]
    right_fitx = right_fit[0] * y ** 2 + right_fit[1] * y + right_fit[2]
    offset_x = (left_fitx + right_fitx - image_size[1]) / 2 * X_METER_PER_PIXEL
    return offset_x


# Draw information overlay containing curve and offset
def draw_info(img, curve_rad,  car_offset):
    rad_text = "{}: {:.2f}m, {:.2f}m".format('Curve values', curve_rad[0], curve_rad[1])
    off_text = "{}: {:.2f}m".format('Car offset', car_offset)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, rad_text, (20, 70), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(img, off_text, (20, 120), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
    return img


# Checks whether the calculates lane lines are plausible
def are_lines_valid(img, lane_lines):
    if lane_lines is None:
        return False
    left_fit = lane_lines[0]
    right_fit = lane_lines[1]

    ploty = np.linspace(0, img.shape[0] - 1, num=img.shape[0])  # to cover same y-range as image

    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

    delta_fitx = right_fitx - left_fitx
    delta_mean = np.mean(delta_fitx)
    delta_min = np.min(delta_fitx)
    delta_max = np.max(delta_fitx)
    if delta_mean > 3.7 / X_METER_PER_PIXEL * 1.2:
        return False
    if delta_mean < 1.5 / X_METER_PER_PIXEL * 0.9:
        return False
    if delta_min < delta_mean * 0.5:
        return False
    if delta_max > delta_mean * 1.5:
        return False
    return True
