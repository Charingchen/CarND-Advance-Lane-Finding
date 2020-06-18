import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
# glob is used to reading all the similar calbration image
import glob


def camera_cali():
    # Arrays to store object points and image points from all images
    objpoints = []
    imgpoints = []

    # Obj points should not change and only based on the chesss board format
    # Preparing object points, like (0,0,0), (1,0,0) ...
    objp = np.zeros((6 * 9, 3), np.float32)

    objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)  # Creating x y coordinates
    # import all cal images
    cal_images = glob.glob('../camera_cal/calibration*.jpg')

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
    return mtx, dist


def mask_image(img, mask_top_width=145, x_offset=20, y_offset=40, dst_offset=200):
    imshape = img.shape
    # Calculate mask height
    img_y_mid = imshape[0] * 0.5
    mask_height = int(img_y_mid * 1.25)
    img_x_mid = int(imshape[1] * 0.5)
    top_left = [img_x_mid - mask_top_width * 0.5, mask_height]
    top_right = [img_x_mid + mask_top_width * 0.5, mask_height]
    bottom_left = [x_offset, imshape[0] - y_offset]
    bottom_right = [imshape[1] - x_offset, imshape[0] - y_offset]

    # Define the source points
    src = np.float32([bottom_left, top_left, top_right, bottom_right])
    # Define destination points
    # 25 is hardcoded to move the lane line to the bottom of the image
    dst = np.float32([[dst_offset, imshape[0] - y_offset + 25], [dst_offset, y_offset],
                      [imshape[1] - dst_offset, y_offset],
                      [imshape[1] - dst_offset, imshape[0] - y_offset + 25]])
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


cali = []
"""
Function to undistort image
Only run Camera calibration once and stored in cali
"""


def undistort_img(img):
    global cali
    if not cali:
        cali = camera_cali()
        # mtx = cali[0]
        # dist = cali[1]
    return cv2.undistort(img, cali[0], cali[1], None, cali[0])


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


def image_process(image, overdrive=False):
    """
    Image process takes a frame image and return a binary image as an input of pipeline logic
    Input:
        image: a RGB image array
        overdrive: boolean True mean complicated computing include all threshold methods,
                   False means simple computing only S threshold and xy gradients
    Return:
        combined_binary: all the threshold binary combined
        color_binary: used to visualize the two combines binary. Debugging use only
    """
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

        # Create color binary to visualize the logic combining
        color_binary = np.dstack((binary_r, color_thresh_binary, gradient_binary)) * 255
        # Combine all the binary
        combined_binary[(color_thresh_binary == 1) | (gradient_binary == 1)] = 1

    else:
        # Grab gradient result
        gradient_binary = sobel_gradient(image, overdrive=overdrive, abs_thresh=abs_thresh)
        combined_binary[(binary_s == 1) | (gradient_binary == 1)] = 1
        color_binary = np.dstack((np.zeros_like(gradient_binary), binary_s, gradient_binary)) * 255

    return combined_binary, color_binary

def binary_wrap_img(undist, overdrive=False, debug=False):
    src, dst = mask_image(undist)

    binary, color_binary = image_process(undist, overdrive)

    img_size = (undist.shape[1], undist.shape[0])

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

