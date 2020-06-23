import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
# glob is used to reading all the similar calibration image
import glob

from frame_process import *


# Define a class to receive the characteristics of each line detection
class Line():
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False
        # x values of the last n fits of the line\
        self.recent_xfitted = []
        # average x values of the fitted line over the last n iterations
        self.bestx = None
        # polynomial coefficients averaged over the last n iterations
        self.best_fit = None
        # polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]
        # radius of curvature of the line in some units
        self.radius_of_curvature = None
        # Confident level, calculated after find the line poly
        self.confident = 0
        # Fail to detect line compare previous result
        self.fail_count = 0

        # Not used yet
        # distance in meters of vehicle center from the line
        self.line_base_pos = None
        # difference in fit coefficients between last and new fits
        self.diffs = np.array([0, 0, 0], dtype='float')
        # x values for detected line pixels
        self.allx = None
        # y values for detected line pixels
        self.ally = None


def fit_one_line(img_shape, y, x, ploty):
    """
    :param img_shape: process image shape
    :param y: collection of y coordinates
    :param x: collection of x coordinates
    :param ploty: A uniformed y coordinates for plotting
    :return: fitted x for easy plotting
            and polynomial coefficient both in pixel and meter
    """
    # it a second order polynomial to each with np.polyfit()
    fit = np.polyfit(y, x, 2)

    # Calcuated X coordinate using ploynomial coefficient and ploty
    fitx = fit[0] * ploty ** 2 + fit[1] * ploty + fit[2]

    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 3 / 80
    xm_per_pix = 3.7 / 570

    # calculate polynomials in meters
    fit_cr = np.polyfit(y * ym_per_pix, x * xm_per_pix, 2)

    # Define y-value where we want radius of curvature
    # We'll choose the maximum y-value, corresponding to the bottom of the image
    y_eval = img_shape[0]

    # Implement the calculation of R_curve (radius of curvature)
    curve_rad = (1 + (2 * fit_cr[0] * y_eval * ym_per_pix + fit_cr[1]) ** 2) ** 1.5 / np.absolute(
        2 * fit_cr[0])
    return fitx, fit, curve_rad


def fit_poly(img_shape, leftx, lefty, rightx, righty):
    # it a second order polynomial to each with np.polyfit()
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    # Generate x and y values for plotting
    ploty = np.linspace(0, img_shape[0] - 1, img_shape[0])
    # Calc both polynomials using ploty, left_fit and right_fit
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 3 / 80  # meters per pixel in y dimension 3/80
    xm_per_pix = 3.7 / 570  # meters per pixel in x dimension 3.7/570

    # calculate polynomials in meters
    left_fit_cr = np.polyfit(lefty * ym_per_pix, leftx * xm_per_pix, 2)
    right_fit_cr = np.polyfit(righty * ym_per_pix, rightx * xm_per_pix, 2)

    # Define y-value where we want radius of curvature
    # We'll choose the maximum y-value, corresponding to the bottom of the image
    y_eval = img_shape[0]

    # Implement the calculation of R_curve (radius of curvature)
    left_curverad = (1 + (2 * left_fit_cr[0] * y_eval * ym_per_pix + left_fit_cr[1]) ** 2) ** 1.5 / np.absolute(
        2 * left_fit_cr[0])
    right_curverad = (1 + (2 * right_fit_cr[0] * y_eval * ym_per_pix + right_fit_cr[1]) ** 2) ** 1.5 / np.absolute(
        2 * right_fit_cr[0])

    return left_fitx, right_fitx, ploty, left_fit, right_fit, [left_curverad, right_curverad]


def find_lane_pixels(binary_warped, run_left=False, run_all=False, plot=False):
    # Logic to setup which line to run
    left_run = run_left or run_all
    right_run = not run_left or run_all

    # Create a uniformed y coordinates for plotting
    img_shape = binary_warped.shape
    ploty = np.linspace(0, img_shape[1] - 1, img_shape[0])
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0] // 2:, :], axis=0)
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0] // 2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # HYPERPARAMETERS
    # Choose the number of sliding windows
    nwindows = 9
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50

    # Set height of windows - based on nwindows above and image shape
    window_height = np.int(binary_warped.shape[0] // nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated later for each window in nwindows
    leftx_current = leftx_base
    rightx_current = rightx_base

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window + 1) * window_height
        win_y_high = binary_warped.shape[0] - window * window_height
        if left_run:
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            # Identify the nonzero pixels in x and y within the window ###
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                              (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            # If found > minpix pixels, recenter next window
            # (`right` or `leftx_current`) on their mean position
            if len(good_left_inds) > minpix:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))

        if right_run:
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin

            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                               (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
            right_lane_inds.append(good_right_inds)
            if len(good_right_inds) > minpix:
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    if left_run:
        try:
            left_lane_inds = np.concatenate(left_lane_inds)
        except ValueError:
            # Avoids an error if the above is not implemented fully
            pass

        # Extract left line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        left_fitx, left_fit, left_curve = fit_one_line(img_shape, lefty, leftx, ploty)

    if right_run:
        try:
            right_lane_inds = np.concatenate(right_lane_inds)
        except ValueError:
            # Avoids an error if the above is not implemented fully
            pass
        # Extract right line pixel position
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]
        right_fitx, right_fit, right_curve = fit_one_line(img_shape, righty, rightx, ploty)

    ## Visualization ##
    # Draw the windows on the visualization image if the plot flag is set
    if plot:
        # Create an output image to draw on and visualize the result
        out_img = np.dstack((binary_warped, binary_warped, binary_warped))
        cv2.rectangle(out_img, (win_xleft_low, win_y_low),
                      (win_xleft_high, win_y_high), (0, 255, 0), 2)
        cv2.rectangle(out_img, (win_xright_low, win_y_low),
                      (win_xright_high, win_y_high), (0, 255, 0), 2)
        # Colors in the left and right lane regions
        out_img[lefty, leftx] = [255, 0, 0]
        out_img[righty, rightx] = [0, 0, 255]

        # Plots the left and right polynomials on the lane lines
        plt.plot(left_fitx, ploty, color='yellow')
        plt.plot(right_fitx, ploty, color='yellow')
        plt.imshow(out_img)
    # Logic to return correct variables
    if run_all:
        return left_fitx, right_fitx, ploty, left_fit, right_fit, [left_curve, right_curve]
    else:
        if run_left:
            return left_fitx, ploty, left_fit, left_curve
        else:
            return right_fitx, ploty, right_fit, right_curve


# def find_lane_pixels(binary_warped, plot=False):
#     # Take a histogram of the bottom half of the image
#     histogram = np.sum(binary_warped[binary_warped.shape[0] // 2:, :], axis=0)
#     # Find the peak of the left and right halves of the histogram
#     # These will be the starting point for the left and right lines
#     midpoint = np.int(histogram.shape[0] // 2)
#     leftx_base = np.argmax(histogram[:midpoint])
#     rightx_base = np.argmax(histogram[midpoint:]) + midpoint
#
#     # HYPERPARAMETERS
#     # Choose the number of sliding windows
#     nwindows = 9
#     # Set the width of the windows +/- margin
#     margin = 100
#     # Set minimum number of pixels found to recenter window
#     minpix = 50
#
#     # Set height of windows - based on nwindows above and image shape
#     window_height = np.int(binary_warped.shape[0] // nwindows)
#     # Identify the x and y positions of all nonzero pixels in the image
#     nonzero = binary_warped.nonzero()
#     nonzeroy = np.array(nonzero[0])
#     nonzerox = np.array(nonzero[1])
#     # Current positions to be updated later for each window in nwindows
#     leftx_current = leftx_base
#     rightx_current = rightx_base
#
#     # Create empty lists to receive left and right lane pixel indices
#     left_lane_inds = []
#     right_lane_inds = []
#
#     # Step through the windows one by one
#     for window in range(nwindows):
#         # Identify window boundaries in x and y (and right and left)
#         win_y_low = binary_warped.shape[0] - (window + 1) * window_height
#         win_y_high = binary_warped.shape[0] - window * window_height
#         # Find the four below boundaries of the window #
#         win_xleft_low = leftx_current - margin
#         win_xleft_high = leftx_current + margin
#         win_xright_low = rightx_current - margin
#         win_xright_high = rightx_current + margin
#
#         # Identify the nonzero pixels in x and y within the window ###
#         good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
#                           (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
#         good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
#                            (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
#
#         # Append these indices to the lists
#         left_lane_inds.append(good_left_inds)
#         right_lane_inds.append(good_right_inds)
#
#         # If found > minpix pixels, recenter next window
#         # (`right` or `leftx_current`) on their mean position
#         if len(good_left_inds) > minpix:
#             leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
#         if len(good_right_inds) > minpix:
#             rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
#
#     # Concatenate the arrays of indices (previously was a list of lists of pixels)
#     try:
#         left_lane_inds = np.concatenate(left_lane_inds)
#         right_lane_inds = np.concatenate(right_lane_inds)
#     except ValueError:
#         # Avoids an error if the above is not implemented fully
#         pass
#
#     # Extract left and right line pixel positions
#     leftx = nonzerox[left_lane_inds]
#     lefty = nonzeroy[left_lane_inds]
#     rightx = nonzerox[right_lane_inds]
#     righty = nonzeroy[right_lane_inds]
#
#     # Fit new polynomials
#     left_fitx, right_fitx, ploty, left_fit, right_fit, curvatures = fit_poly(binary_warped.shape, leftx, lefty, rightx,
#                                                                              righty)
#
#     ## Visualization ##
#     # Draw the windows on the visualization image if the plot flag is set
#     if plot:
#         # Create an output image to draw on and visualize the result
#         out_img = np.dstack((binary_warped, binary_warped, binary_warped))
#         cv2.rectangle(out_img, (win_xleft_low, win_y_low),
#                       (win_xleft_high, win_y_high), (0, 255, 0), 2)
#         cv2.rectangle(out_img, (win_xright_low, win_y_low),
#                       (win_xright_high, win_y_high), (0, 255, 0), 2)
#         # Colors in the left and right lane regions
#         out_img[lefty, leftx] = [255, 0, 0]
#         out_img[righty, rightx] = [0, 0, 255]
#
#         # Plots the left and right polynomials on the lane lines
#         plt.plot(left_fitx, ploty, color='yellow')
#         plt.plot(right_fitx, ploty, color='yellow')
#         plt.imshow(out_img)
#
#     return left_fitx, right_fitx, ploty, left_fit, right_fit, curvatures
def search_around_poly(binary_warped, run_left=False, run_all=False, plot=False):
    # HYPERPARAMETER
    # Choose the width of the margin around the previous polynomial to search
    # The quiz grader expects 100 here, but feel free to tune on your own!
    margin = 100
    # Create a uniformed y coordinates for plotting
    img_shape = binary_warped.shape
    ploty = np.linspace(0, img_shape[1] - 1, img_shape[0])

    # Logic to setup which line to run

    left_run = run_left or run_all
    right_run = not run_left or run_all

    # Grab activated pixels
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    if left_run:
        left_lane_inds = ((nonzerox >= (left_line.current_fit[0] * nonzeroy ** 2 + left_line.current_fit[1] * nonzeroy +
                                        left_line.current_fit[2] - margin)) &
                          (nonzerox < (left_line.current_fit[0] * nonzeroy ** 2 + left_line.current_fit[1] * nonzeroy +
                                       left_line.current_fit[2] + margin))).nonzero()[0]
        # Again, extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        left_fitx, left_fit, left_curve = fit_one_line(img_shape, lefty, leftx, ploty)

    if right_run:
        right_lane_inds = ((nonzerox >= (right_line.current_fit[0] * nonzeroy ** 2 + right_line.current_fit[1] * nonzeroy +
                                         right_line.current_fit[2] - margin)) &
                           (nonzerox < (right_line.current_fit[0] * nonzeroy ** 2 + right_line.current_fit[1] * nonzeroy +
                                        right_line.current_fit[2] + margin))).nonzero()[0]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]
        right_fitx, right_fit, right_curve = fit_one_line(img_shape, righty, rightx, ploty)


    ## Visualization ##
    # Create an image to draw on and an image to show the selection window
    if plot:
        out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
        window_img = np.zeros_like(out_img)
        # Color in left and right line pixels
        out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

        # Generate a polygon to illustrate the search window area
        # And recast the x and y points into usable format for cv2.fillPoly()
        left_line_window1 = np.array([np.transpose(np.vstack([left_fitx - margin, ploty]))])
        left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx + margin,
                                                                        ploty])))])
        left_line_pts = np.hstack((left_line_window1, left_line_window2))
        right_line_window1 = np.array([np.transpose(np.vstack([right_fitx - margin, ploty]))])
        right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx + margin,
                                                                         ploty])))])
        right_line_pts = np.hstack((right_line_window1, right_line_window2))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(window_img, np.int_([left_line_pts]), (0, 255, 0))
        cv2.fillPoly(window_img, np.int_([right_line_pts]), (0, 255, 0))
        result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)

        # Plot the polynomial lines onto the image
        plt.plot(left_fitx, ploty, color='yellow')
        plt.plot(right_fitx, ploty, color='yellow')

        plt.imshow(result)

    # Logic to return correct variables
    if run_all:
        return left_fitx, right_fitx, ploty, left_fit, right_fit, [left_curve, right_curve]
    else:
        if run_left:
            return left_fitx, ploty, left_fit, left_curve
        else:
            return right_fitx, ploty, right_fit, right_curve


#
# def search_around_poly(binary_warped, left_fit, right_fit, plot=False):
#     # HYPERPARAMETER
#     # Choose the width of the margin around the previous polynomial to search
#     # The quiz grader expects 100 here, but feel free to tune on your own!
#     margin = 100
#
#     # Grab activated pixels
#     nonzero = binary_warped.nonzero()
#     nonzeroy = np.array(nonzero[0])
#     nonzerox = np.array(nonzero[1])
#
#     ### TO-DO: Set the area of search based on activated x-values ###
#     ### within the +/- margin of our polynomial function ###
#     ### Hint: consider the window areas for the similarly named variables ###
#     ### in the previous quiz, but change the windows to our new search area ###
#
#     left_lane_inds = ((nonzerox >= (left_fit[0] * nonzeroy ** 2 + left_fit[1] * nonzeroy +
#                                     left_fit[2] - margin)) &
#                       (nonzerox < (left_fit[0] * nonzeroy ** 2 + left_fit[1] * nonzeroy +
#                                    left_fit[2] + margin))).nonzero()[0]
#     right_lane_inds = ((nonzerox >= (right_fit[0] * nonzeroy ** 2 + right_fit[1] * nonzeroy +
#                                      right_fit[2] - margin)) &
#                        (nonzerox < (right_fit[0] * nonzeroy ** 2 + right_fit[1] * nonzeroy +
#                                     right_fit[2] + margin))).nonzero()[0]
#
#     # Again, extract left and right line pixel positions
#     leftx = nonzerox[left_lane_inds]
#     lefty = nonzeroy[left_lane_inds]
#     rightx = nonzerox[right_lane_inds]
#     righty = nonzeroy[right_lane_inds]
#
#     # Fit new polynomials
#     left_fitx, right_fitx, ploty, left_fit, right_fit, curvatures = fit_poly(binary_warped.shape, leftx, lefty, rightx,
#                                                                              righty)
#
#     ## Visualization ##
#     # Create an image to draw on and an image to show the selection window
#     if plot:
#         out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
#         window_img = np.zeros_like(out_img)
#         # Color in left and right line pixels
#         out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
#         out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
#
#         # Generate a polygon to illustrate the search window area
#         # And recast the x and y points into usable format for cv2.fillPoly()
#         left_line_window1 = np.array([np.transpose(np.vstack([left_fitx - margin, ploty]))])
#         left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx + margin,
#                                                                         ploty])))])
#         left_line_pts = np.hstack((left_line_window1, left_line_window2))
#         right_line_window1 = np.array([np.transpose(np.vstack([right_fitx - margin, ploty]))])
#         right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx + margin,
#                                                                          ploty])))])
#         right_line_pts = np.hstack((right_line_window1, right_line_window2))
#
#         # Draw the lane onto the warped blank image
#         cv2.fillPoly(window_img, np.int_([left_line_pts]), (0, 255, 0))
#         cv2.fillPoly(window_img, np.int_([right_line_pts]), (0, 255, 0))
#         result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
#
#         # Plot the polynomial lines onto the image
#         plt.plot(left_fitx, ploty, color='yellow')
#         plt.plot(right_fitx, ploty, color='yellow')
#
#         plt.imshow(result)
#
#     return left_fitx, right_fitx, ploty, left_fit, right_fit, curvatures


def draw_poly_fill(binary_wrap, undist, left_fitx, right_fitx, ploty, curvatures):
    # Draw unwarped the poly fill onto the image
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(binary_wrap).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = perspective_transform(color_warp, inverse=True)
    # cv2.warpPerspective(color_warp, Minv, (image.shape[1], image.shape[0]))
    # Combine the result with the original image
    result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)

    left_curve_text = 'Left Radius of Curvature: ' + str(round(curvatures[0], 2)) + 'Meters'
    right_curve_text = 'Right Radius of Curvature: ' + str(round(curvatures[1], 2)) + 'Meters'
    #     print(curvatures[0],curvatures[1])
    result = cv2.putText(result, left_curve_text, (200, 100), cv2.FONT_HERSHEY_SIMPLEX,
                         1, (0, 255, 255), 2, cv2.LINE_AA)
    result = cv2.putText(result, right_curve_text, (200, 150), cv2.FONT_HERSHEY_SIMPLEX,
                         1, (0, 255, 255), 2, cv2.LINE_AA)
    return result


def single_lane_detection(line,binary_wrap,undist,run_left=False,fail_counter = 5):
    # If the line detected previous iteration, preform search base on previous polynomial
    if line.detected:
        #binary_wrap = binary_wrap_img(undist)
        fitx, ploty,fit,curve = search_around_poly(binary_wrap,run_left)
    else:
        if line.fail_count < fail_counter:
        # If fail counter is less, preform simplify image filter and use sliding window to find lines
            #binary_wrap = binary_wrap_img(undist)
            fitx,ploty,fit,curve = find_lane_pixels(binary_wrap,run_left)
        # If fail too many times, preform a more complex image filter and sliding window
        else:
            new_binary_wrap = binary_wrap_img(undist, overdrive=True)
            # Need to find a way to share this in case of the other line failed to detect as well
            fitx, ploty, fit, curve = find_lane_pixels(new_binary_wrap, run_left)

    # Implement confident level, relate this with previous confident level to reduce check






# def video_lane_detectoion(img):
#     global left_line, right_line
#     fail_allowed = 5
#     threshold = 0.5  # 5% threshold
#
#     # Un-distort image using Camera calibration data
#     undist = undistort_img(img)
#
#     # Left line detection
#
#     # Right line detection






def video_lane_detectoion(img):
    global fail_counter
    global left_line, right_line
    fail_allowed = 5
    threshold = 0.5  # 5% threshold

    # Undistort image using Camera calibration data
    undist = undistort_img(img)

    # Check if previou line is detected
    if left_line.detected and right_line.detected:
        # Set overdrive to defalut False to reduce runtime
        binary_wrap = binary_wrap_img(undist)
        left_fitx, right_fitx, ploty, left_fit, right_fit, curvatures = \
            search_around_poly(binary_wrap, left_line.current_fit, right_line.current_fit)
    else:
        # If fail counter is < 5 times, preform quick filter and calcuations
        if fail_counter < fail_allowed:
            binary_wrap = binary_wrap_img(undist)
            left_fitx, right_fitx, ploty, left_fit, right_fit, curvatures = \
                find_lane_pixels(binary_wrap)
        # If fail counter is > 5 times, preform full on filter
        else:
            binary_wrap = binary_wrap_img(undist, overdrive=True)
            left_fitx, right_fitx, ploty, left_fit, right_fit, curvatures = \
                find_lane_pixels(binary_wrap)

    confident_level = 0

    # Check empty first, if they are none meaning first time runing,
    # plot the fill right away
    if left_line.radius_of_curvature and right_line.radius_of_curvature:

        # Check the new lines against previous curvatures
        if curvatures[0] <= left_line.radius_of_curvature * (1 + threshold) and \
                curvatures[0] > left_line.radius_of_curvature * (1 - threshold) and \
                curvatures[1] <= right_line.radius_of_curvature * (1 + threshold) and \
                curvatures[1] > right_line.radius_of_curvature * (1 - threshold):
            confident_level += 1

        # Check the new lines if they seperate about the same compare to average
        current_x_dist = right_fitx[0] - left_fitx[0]
        before_x_dist = right_line.recent_xfitted[0] - left_line.recent_xfitted[0]
        if current_x_dist <= before_x_dist * (1 + threshold) and current_x_dist > before_x_dist * (1 - threshold):
            confident_level += 1
        # Check the slope from previouse to see if the new result is approx parallel
        # Slope of second degree ploy is its derivative, which is Ax + B

        # Calculate slope for the current and last poly fit lines
        left_slope = left_fit[0] * ploty + left_fit[1]
        right_slope = right_fit[0] * ploty + right_fit[1]
        prev_left_slope = left_line.current_fit[0] * ploty + left_line.current_fit[1]
        prev_right_slope = right_line.current_fit[0] * ploty + right_line.current_fit[1]

        # Compare every element in slope to see if the two poly is parallel
        left_hit = 0
        right_hit = 0
        for i in range(len(left_slope)):
            if left_slope[i] > prev_left_slope[i] * (1 - threshold) and \
                    left_slope[i] <= prev_left_slope[i] * (1 + threshold):
                left_hit += 1
            if right_slope[i] > prev_right_slope[i] * (1 - threshold) and \
                    right_slope[i] <= prev_right_slope[i] * (1 + threshold):
                right_hit += 1

        if left_hit / len(left_slope) >= 0.8 and right_hit / len(right_slope) >= 0.8:
            confident_level += 1

        # If condfident level high,append to previous averaging
        if confident_level >= 3:
            # Record everything to the line classs
            left_line.current_fit = left_fit
            right_line.current_fit = right_fit

            left_line.radius_of_curvature = curvatures[0]
            right_line.radius_of_curvature = curvatures[1]

            left_line.detected = True
            right_line.detected = True

            left_fitx_temp = [left_line.recent_xfitted, left_fitx]
            right_fitx_temp = [right_line.recent_xfitted, right_fitx]

            # calculate best fitx for both line in case there is a detection fail in the code
            left_line.bestx = np.mean(left_fitx_temp, axis=0)
            right_line.bestx = np.mean(right_fitx_temp, axis=0)

            fail_counter = 0
        else:
            # If not use previous average result and count fail detected and flag line detect false
            left_line.detected = False
            right_line.detected = False
            confident_level = 0
            fail_counter += 1
            left_fitx = left_line.bestx
            right_fitx = right_line.bestx

    else:
        left_line.detected = True
        right_line.detected = True
        left_line.radius_of_curvature = curvatures[0]
        right_line.radius_of_curvature = curvatures[1]
        left_line.current_fit = left_fit
        right_line.current_fit = right_fit
        left_line.recent_xfitted = left_fitx
        right_line.recent_xfitted = right_fitx
        left_line.bestx = left_fitx
        right_line.bestx = right_fitx

    result_img = draw_poly_fill(binary_wrap, undist, left_fitx, right_fitx, ploty, curvatures)
    # print('Confident Level:',confident_level)
    # print('fail Counter:', fail_counter)
    return result_img


from moviepy.editor import VideoFileClip
from IPython.display import HTML

left_line = Line()
right_line = Line()
fail_counter = 0
output = '../temp_output/video_output/project_video.mp4'
## To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
## To do so add .subclip(start_second,end_second) to the end of the line below
## Where start_second and end_second are integer values representing the start and end of the subclip
## You may also uncomment the following line for a subclip of the first 5 seconds
# clip2 = VideoFileClip('../project_video.mp4')
clip2 = VideoFileClip('../project_video.mp4').subclip(0, 5)
project_clip = clip2.fl_image(video_lane_detectoion)
project_clip.write_videofile(output, audio=False)
