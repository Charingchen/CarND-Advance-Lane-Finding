import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
#glob is used to reading all the similar calbration image
import glob

# Arrays to store object points and image points from all images
objpoints = []
imgpoints = []

# Obj points should not change and only based on the chesss board format
# Preparing object points, like (0,0,0), (1,0,0) ...
objp = np.zeros((6 * 9, 3), np.float32)

objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)  # Creating x y coordinates
# import all cal images
cal_images = glob.glob ('../camera_cal/calibration*.jpg')

for fname in cal_images:
    # read in each image
    img = mpimg.imread(fname)

    # Convert to gray scale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find Chesse board corners
    ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)

    if ret:
        imgpoints.append(corners)
        objpoints.append(objp)

# Get the Camera matrix
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints,
                                                   img.shape[1:], None, None)
def mask_image (img,mask_top_width = 145, x_offset = 20, y_offset=40, dst_offset = 200):

    imshape=img.shape
    # Calculate mask height
    img_y_mid = imshape[0]*0.5
    mask_height = int(img_y_mid*1.25)
    img_x_mid = int(imshape[1]*0.5)
    top_left = [img_x_mid - mask_top_width*0.5 , mask_height]
    top_right = [img_x_mid + mask_top_width*0.5 , mask_height]
    bottom_left = [x_offset,imshape[0]-y_offset]
    bottom_right = [imshape[1]-x_offset,imshape[0]-y_offset]

    # Define the source points
    src = np.float32([bottom_left,top_left,top_right,bottom_right])
    # Define destination points
    # 25 is hardcoded to move the lane line to the bottom of the image
    dst = np.float32([[dst_offset, imshape[0]-y_offset+25], [dst_offset, y_offset],
                                     [imshape[1]-dst_offset, y_offset],
                                     [imshape[1]-dst_offset, imshape[0]-y_offset+25]])
    return src, dst


def perspective_transform(undist, inverse=False, debug=False):
    src, dst = mask_image(undist)
    if inverse == False:
        M = cv2.getPerspectiveTransform(src, dst)
    else:
        M = cv2.getPerspectiveTransform(dst, src)

    img_size = (undist.shape[1], undist.shape[0])
    warped = cv2.warpPerspective(undist, M, img_size)

    if debug:
        return warped, src, dst
    else:
        return warped


# Function use to undistort images
def undistort_img(img, mtx=mtx, dist=dist):
    return cv2.undistort(img, mtx, dist, None, mtx)


def abs_sobel_thresh(gray, orient='x', thresh=(0, 255)):
    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0))
    if orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1))

    scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))
    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
    return binary_output


def mag_threshold(gray, sobel_kernel=3, thresh=(0, 255)):
    # Take both Sobel x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Calculate the gradient magnitude
    gradmag = np.sqrt(sobelx ** 2 + sobely ** 2)
    # Rescale to 8 bit
    scale_factor = np.max(gradmag) / 255
    gradmag = (gradmag / scale_factor).astype(np.uint8)
    # Create a binary image of ones where threshold is met, zeros otherwise
    binary_output = np.zeros_like(gradmag)
    binary_output[(gradmag >= thresh[0]) & (gradmag <= thresh[1])] = 1

    return binary_output


def dir_threshold(gray, sobel_kernel=3, thresh=(0, np.pi / 2)):
    # Take the gradient in x and y separately
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Take the absolute value of the x and y gradients
    abs_sobelx = np.absolute(sobelx)
    abs_sobely = np.absolute(sobely)
    # Use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the gradient
    abs_arctan = np.arctan2(abs_sobely, abs_sobelx)
    # Create a binary mask where direction thresholds are met
    binary_output = np.zeros_like(abs_arctan)
    binary_output[(abs_arctan >= thresh[0]) & (abs_arctan <= thresh[1])] = 1
    # Return this mask as your binary_output image
    return binary_output


def sobel_gradient(image, overdrive=True, ksize=3, abs_thresh=(25, 200),
                   mag_thresh=(40, 150), dir_thresh=(0.7, 1.3)):
    # Convert to gray scale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # Apply each of the thresholding functions according to the compliexity
    gradx = abs_sobel_thresh(gray, orient='x', thresh=abs_thresh)
    grady = abs_sobel_thresh(gray, orient='y', thresh=abs_thresh)
    combined = np.zeros_like(grady)

    # Choose complexity True means more complicated calculation
    # If min complexity require only execute x and y gradient
    if overdrive:
        mag_binary = mag_threshold(gray, sobel_kernel=ksize, thresh=mag_thresh)
        dir_binary = dir_threshold(gray, sobel_kernel=ksize, thresh=dir_thresh)
        combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1
    else:
        combined[(gradx == 1) & (grady == 1)] = 1

    return combined


# A function only process h and s channels
def convert_to_hs(image, thresh_h=[18, 100], thresh_s=[90, 255]):
    # First convert image to HLS and only take H and S
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    H = hls[:, :, 0]
    S = hls[:, :, 2]
    # Create Binary arrays
    binary_h = np.zeros_like(H)
    binary_s = np.zeros_like(S)

    binary_h[(H > thresh_h[0]) & (H <= thresh_h[1])] = 1
    binary_s[(S > thresh_s[0]) & (S <= thresh_s[1])] = 1

    return binary_h, binary_s


"""
Image process takes a frame iamge and return a binary image as an input of pipeline logic
Input: 
    image: a RGB image array 
    overdrive: boolean True mean complicated computing include all threshold methods, 
               False means simple computing only S threshold and xy graidents
Return:
    combined_binary: all the threhold binary combined
    color_binary: used to visuallize the two combines binary. Debugging use only
"""


def image_process(image, overdrive=False):
    # Fine tune all the threshold here
    thresh_h = [23, 100]
    thresh_s = [170, 255]
    thresh_r = [220, 255]
    abs_thresh = (50, 200)
    mag_thresh = (60, 150)

    # Get binary H and S from hls color space
    binary_h, binary_s = convert_to_hs(image, thresh_h=thresh_h, thresh_s=thresh_s)
    combined_binary = np.zeros_like(binary_s)

    if overdrive:
        # Grab gradient result
        gradient_binary = sobel_gradient(image, ksize=15,
                                         abs_thresh=abs_thresh, mag_thresh=mag_thresh)

        # Implement R threhold
        R = image[:, :, 0]

        binary_r = np.zeros_like(R)
        binary_r[(R > thresh_r[0]) & (R <= thresh_r[1])] = 1

        # Combine R with H and S threshold binary
        color_thresh_binary = np.zeros_like(binary_s)
        color_thresh_binary[((binary_h == 1) & (binary_s == 1)) | (binary_r == 1)] = 1

        # Create color binary to visualiize the logic combining
        color_binary = np.dstack((binary_r, color_thresh_binary, gradient_binary)) * 255
        # Combine all the binary
        combined_binary[(color_thresh_binary == 1) | (gradient_binary == 1)] = 1

    else:
        # Grab gradient result
        gradient_binary = sobel_gradient(image, overdrive=overdrive, abs_thresh=abs_thresh)
        combined_binary[(binary_s == 1) | (gradient_binary == 1)] = 1
        color_binary = np.dstack((np.zeros_like(gradient_binary), binary_s, gradient_binary)) * 255

    return combined_binary, color_binary


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


def find_lane_pixels(binary_warped, plot=False):
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
        # Find the four below boundaries of the window #
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin

        # Identify the nonzero pixels in x and y within the window ###
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                          (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                           (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        # If found > minpix pixels, recenter next window
        # (`right` or `leftx_current`) on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices (previously was a list of lists of pixels)
    try:
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
    except ValueError:
        # Avoids an error if the above is not implemented fully
        pass

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit new polynomials
    left_fitx, right_fitx, ploty, left_fit, right_fit, curvatures = fit_poly(binary_warped.shape, leftx, lefty, rightx,
                                                                             righty)

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

    return left_fitx, right_fitx, ploty, left_fit, right_fit, curvatures


def search_around_poly(binary_warped, left_fit, right_fit, plot=False):
    # HYPERPARAMETER
    # Choose the width of the margin around the previous polynomial to search
    # The quiz grader expects 100 here, but feel free to tune on your own!
    margin = 100

    # Grab activated pixels
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    ### TO-DO: Set the area of search based on activated x-values ###
    ### within the +/- margin of our polynomial function ###
    ### Hint: consider the window areas for the similarly named variables ###
    ### in the previous quiz, but change the windows to our new search area ###

    left_lane_inds = ((nonzerox >= (left_fit[0] * nonzeroy ** 2 + left_fit[1] * nonzeroy +
                                    left_fit[2] - margin)) &
                      (nonzerox < (left_fit[0] * nonzeroy ** 2 + left_fit[1] * nonzeroy +
                                   left_fit[2] + margin))).nonzero()[0]
    right_lane_inds = ((nonzerox >= (right_fit[0] * nonzeroy ** 2 + right_fit[1] * nonzeroy +
                                     right_fit[2] - margin)) &
                       (nonzerox < (right_fit[0] * nonzeroy ** 2 + right_fit[1] * nonzeroy +
                                    right_fit[2] + margin))).nonzero()[0]

    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit new polynomials
    left_fitx, right_fitx, ploty, left_fit, right_fit, curvatures = fit_poly(binary_warped.shape, leftx, lefty, rightx,
                                                                             righty)

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

    return left_fitx, right_fitx, ploty, left_fit, right_fit, curvatures



"""
Logic to handle video processing frame by frame find lan pixel

"""
def fit_lane_line(binary_wrap, prev_fit, plot=False):
    # Detect if there is previous line fit
    if not prev_fit:
        # call find lane line for the first time
        left_fitx, right_fitx, ploty, left_fit, right_fit, curvatures = find_lane_pixels(binary_wrap, plot)
    else:
        left_fitx, right_fitx, ploty, left_fit, right_fit, curvatures = search_around_poly(binary_wrap, prev_fit[0],
                                                                                           prev_fit[1], plot)
        # call search around poly instead
        # search around poly has handle there is no find according to prevous logic to catch
        # search find failure
    prev_fit = [left_fit, right_fit]

    return left_fitx, right_fitx, ploty, prev_fit, curvatures


# Define a class to receive the characteristics of each line detection
class Line():
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False
        # x values of the last n fits of the line
        self.recent_xfitted = []
        # average x values of the fitted line over the last n iterations
        self.bestx = None
        # polynomial coefficients averaged over the last n iterations
        self.best_fit = None
        # polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]
        # radius of curvature of the line in some units
        self.radius_of_curvature = None

        # Not used yet
        # distance in meters of vehicle center from the line
        self.line_base_pos = None
        # difference in fit coefficients between last and new fits
        self.diffs = np.array([0, 0, 0], dtype='float')
        # x values for detected line pixels
        self.allx = None
        # y values for detected line pixels
        self.ally = None


def draw_poly_fill(binary_wrap, undist, left_fitx, right_fitx, ploty,curvatures):
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


def binary_wrap_img(undist, overdrive=False, debug=False):
    src, dst = mask_image(undist)

    binary, color_binary = image_process(undist, overdrive)

    img_size = (img.shape[1], img.shape[0])

    warped = perspective_transform(binary)
    if debug:
        f = plt.figure()
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
        f.tight_layout()
        ax1.imshow(binary)
        ax1.set_title('Binary Image', fontsize=50)
        ax2.imshow(warped)
        ax2.set_title('Warped Image', fontsize=50)
        plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)

    return warped

def video_lane_detectoion(img):
    global fail_counter
    global left_line, right_line
    fail_allowed = 5
    threshold = 0.5  # 5% threshold

    # Undistort image using Camera calibration data
    undist = cv2.undistort(img, mtx, dist, None, mtx)

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

            left_fitx_temp = [left_line.recent_xfitted,left_fitx]
            right_fitx_temp = [right_line.recent_xfitted,right_fitx]

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

    result_img = draw_poly_fill(binary_wrap, undist, left_fitx, right_fitx, ploty,curvatures)
    print('Confident Level:',confident_level)
    print('fail Counter:', fail_counter)
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
##clip2 = VideoFileClip('test_videos/solidYellowLeft.mp4').subclip(0,5)
clip2 = VideoFileClip('../project_video.mp4')
project_clip = clip2.fl_image(video_lane_detectoion)
project_clip.write_videofile(output, audio=False)