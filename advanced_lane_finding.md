# Advanced Lane Finding Project #


The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

---------------------------------


## Camera calibration and distortion coefficients acquisition ##
	
This is a common method for the acquisition of camera distortion coefficients, please pay attention to the count the number of chessboard corners.
Here is 9x6. After the first runing of these code, the distortion coefficients  will be stored in pickle files.

	import numpy as np
	import cv2
	import glob
	import matplotlib.pyplot as plt
	import pickle

	def get_camera_parm():

	    # chessboard corners 6 * 9
	    objp = np.zeros((6 * 9, 3), np.float32)
	    objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)
	
	    objpoints = []  # 3d points in real world space
	    imgpoints = []  # 2d points in image plane
	
	    images = glob.glob('camera_cal/calibration*.jpg')
	
	    f, axes = plt.subplots(1, 2, figsize=(30, 30))
	
	    # read 20 images and draw chessboard corners
	    for index, iname in enumerate(images):
	        img = cv2.imread(iname)
	        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	        ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)
	
	        if (index == 0):
	            # Plotting the original Image
	            axes[0].set_title('Original Image', fontsize=20)
	            axes[0].imshow(img)
	
	        if ret:
	            objpoints.append(objp)
	            imgpoints.append(corners)
	            cv2.drawChessboardCorners(img, (9, 6), corners, ret)
	
	            if (index == 0):
	                axes[1].set_title('Image with Chessboard Corners', fontsize=20)
	                axes[1].imshow(img)
	
	    # calculate distortion coefficients
	    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
	
	    # Storage camera parameters distortion coefficients to pickle file
	    file_name = 'camera_cal/camera_internal_parm.pickle'
	    cam_parm_dict = {'ret': ret, 'mtx': mtx, 'dist': dist, 'rvecs': rvecs, 'tvecs': tvecs}
	    with open(file_name, 'wb') as fp:
	        pickle.dump(cam_parm_dict, fp, protocol=pickle.HIGHEST_PROTOCOL)


[![nx6SxA.png](https://s2.ax1x.com/2019/09/21/nx6SxA.png)](https://imgchr.com/i/nx6SxA)


## Perspective transform and brids eye view ##

Note that the selection and corresponding of the source and destination points.

	def bird_eye(undistorted_img):
	
	    y = undistorted_img.shape[0]
	    x = undistorted_img.shape[1]
	
	    # Define 4 source points
	    source = np.float32([[490, 482], [810, 482],
	                         [1250, 720], [40, 720]])
	
	    # Define 4 destination points (must be listed in the same order as src points!)
	    destination = np.float32([[0, 0], [1280, 0],
	                              [1250, 720], [40, 720]])
	
	    # calculate Perspective transform matrix
	    M = cv2.getPerspectiveTransform(source, destination)
	    
	    # warp image to a top-down view
	    binary_warped = cv2.warpPerspective(undistorted_img, M, (x, y))
	
	    return binary_warped, M

## Color space and thresholding ##


	def color_filter(image, threshold, channel, color_space):
	    
	    # color space transform
	    color_img = cv2.cvtColor(image, color_space)
	    
	    # select special color channel
	    color_channel = color_img[:, :, channel]
	    binary_img = np.zeros_like(color_channel)
	    binary_img[(color_channel >= threshold[0]) &
	               (color_channel <= threshold[1])] = 1
	    
	    return binary_img

## Combine image masks ##

Use the gradient mask with sobel operator is a complete disaster. It will bring a very big noise. 
HSV color space will also bring many noise.
Thus I use L channel of  LUV color space to detect white line , the B of Lab color space to detect yellow line.

	def mask_combine(image, model=0):
	    binary_warped, M = bird_eye(image)
	
	    hls_s = color_filter(binary_warped, [100, 255], 2, cv2.COLOR_RGB2HLS)  # both white lines and yellow lines
	    luv_l = color_filter(binary_warped, [215, 255], 0, cv2.COLOR_RGB2LUV)  # only white lines
	    lab_b = color_filter(binary_warped, [145, 200], 2, cv2.COLOR_RGB2Lab)  # only yellow lines
	    combined_binary = np.zeros_like(lab_b)
	
	    # model==1 is only used for harder challenge video
	    if model == 1:
	        luv_l[:, 0:640] = 0
	        hls_s[:, 0:640] = 0
	        lab_b[:, 640:1280] = 0
	        combined_binary[(hls_s == 1) | (lab_b == 1)] = 1
	    else:
	        # hls color space is useless in challege video
	        combined_binary[(luv_l == 1) | (lab_b == 1)] = 1
	

## Histogram ##

	histogram = np.sum(image, axis=0)

## Run test_images ##
	
	if __name__ == '__main__':
	    # get_camera_parm()
	
	    images = glob.glob('test_images/*.jpg')
	
	    f, axes = plt.subplots(8, 6, figsize=(15, 30))
	    f.subplots_adjust(hspace=0.5)
	    axes[0, 0].set_title('OriginalImages', fontsize=10)
	    axes[0, 1].set_title('Combined binary images', fontsize=10)
	    axes[0, 2].set_title('S channel of HLS', fontsize=10)
	    axes[0, 3].set_title('L channel of LUV', fontsize=10)
	    axes[0, 4].set_title('B channel of Lab', fontsize=10)
	    axes[0, 5].set_title('Histogram', fontsize=10)
	
	    # read distortion coefficients from pickle file
	    dist_pickle = pickle.load(open("camera_cal/camera_internal_parm.pickle", "rb"))
	    mtx = dist_pickle["mtx"]
	    dist = dist_pickle["dist"]
	
	    for item, originalImage in enumerate(images):
	        ori_img = cv2.imread(originalImage)
	        ori_img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2RGB).astype('uint8')
	
	        undistorted_img = cv2.undistort(ori_img, mtx, dist, None, mtx)
	        combined_binary, hls_s, luv_l, lab_b = mask_combine(undistorted_img)
	        histogram = np.sum(combined_binary, axis=0)
	
	        axes[item, 0].imshow(ori_img)
	        axes[item, 1].imshow(combined_binary, cmap='gray')
	        axes[item, 2].imshow(hls_s, cmap='gray')
	        axes[item, 3].imshow(luv_l, cmap='gray')
	        axes[item, 4].imshow(lab_b, cmap='gray')
	        axes[item, 5].plot(histogram)
	
	    plt.show()

[![nxyz2d.png](https://s2.ax1x.com/2019/09/21/nxyz2d.png)](https://imgchr.com/i/nxyz2d)

## Define Line Class to cache frames and find the line location ##

I defined a Line() class to keep track of all the interesting parameters you measure from frame to frame.
I performed sanity check to separate the left and right line.  I also carriede out the look-ahead filter to search the points quickly. And I also averaged caches of 10 frame or 3 frame to get a smooth results

	from collections import deque
	from moviepy.editor import VideoFileClip
	from pre_processing import *


	class Line:
	    def __init__(self):
	
	        # storage results of 10 frame
	        Max_storage = 10
	        # if the line is detected -> search points in the nearby region,
	        # not -> use histogram to search points in big region
	        self.detected = False
	        self.over_side = False
	        # Fit line intersection value of upper and lower boundaries of the image
	        self.lastx_bottom = None
	        self.lastx_top = None
	        # curvature radius
	        self.radius = None
	
	        # storage polynomial coefficient of 10 frames
	        self.fit_p0 = deque(maxlen=Max_storage)
	        self.fit_p1 = deque(maxlen=Max_storage)
	        self.fit_p2 = deque(maxlen=Max_storage)
	
	        # fit line point x value
	        self.fitx = None
	
	        # storage points
	        self.pts = []
	
	        # storage points x, y value
	        self.X = None
	        self.Y = None
	
	        # storage Fit line intersection x value of upper and lower boundaries of the image of 10 frames
	        self.bottom = deque(maxlen=Max_storage)
	        self.top = deque(maxlen=Max_storage)
	
	        # in order to get averaged value of radius of 3 frame
	        self.count = 0
	
	    def quick_search(self, x, y):
	
	        # search points in the nearby region,
	
	        x_pts = []
	        y_pts = []
	        beyond_border = 0
	
	        if self.detected == True:
	            i = 720
	            j = 630
	
	            while j >= 0:
	                y_pt = np.mean([i, j])
	                x_pt = (np.mean(self.fit_p0)) * y_pt ** 2 + \
	                       (np.mean(self.fit_p1) * y_pt) + (np.mean(self.fit_p2))
	
	                x_id = np.where(
	                    ((x_pt - 25) < x) & (x < (x_pt + 25)) & ((y > j) & (y < i)))
	                x_window, y_window = x[x_id], y[x_id]
	
	                if np.sum(x_window) != 0:
	                    np.append(x_pts, x_window)
	                    np.append(y_pts, y_window)
	                    try:
	                        over_left_edge = x_window.index(0)
	
	                        self.over_side = True
	                        break
	                    except Exception as  e:
	                        pass
	
	                    try:
	                        over_right_edge = x_window.index(1280)
	
	                        self.over_side = True
	                        break
	                    except Exception as e:
	                        pass
	                else:
	                    if self == Left:
	                        hist_thresh = [0, 640]
	                    else:
	                        hist_thresh = [640, 1280]
	                # the height of window is 90
	                j -= 90
	                i -= 90
	
	        # if no useful points, that means we dont detect the line
	        if np.sum(x_pts) == 0:
	            self.detected = False
	
	        return x_pts, y_pts, self.detected, self.over_side
	
	    def slow_search(self, x, y, image):
	
	        x_pts = []
	        y_pts = []
	        if self == Left:
	            hist_thresh = [0, 640]
	        else:
	            hist_thresh = [640, 1280]
	
	        if self.detected == False:
	            i = 720
	            j = 630
	
	            while j >= 0:
	                # the histogram in a small region between y=i and y=j
	                histogram = np.sum(image[j:i, :], axis=0)
	
	                peak_x = np.argmax(histogram[hist_thresh[0]:hist_thresh[1]]) + hist_thresh[0]
	
	                x_id = np.where(
	                    ((peak_x - 25) < x) & (x < (peak_x + 25)) & ((y > j) & (y < i)))
	                x_window, y_window = x[x_id], y[x_id]
	
	                # get index of useful points
	                if np.sum(x_window) != 0:
	                    x_pts.extend(x_window)
	                    y_pts.extend(y_window)
	                    #  Whether the Fit line intersects with the left edge, mid line, or right edge
	                    #  if the fit line intersect, we need change the interval of histogram-peak-find .
	
	                    try:
	                        over_mid = x_window.index(640)
	                        hist_thresh = [560, 720]
	                    except Exception as e:
	                        pass
	
	                    try:
	                        over_left_edge = x_window.index(0)
	                        hist_thresh = [0, 320]
	                        self.over_side = True
	                        break
	                    except Exception as  e:
	                        pass
	
	                    try:
	                        over_right_edge = x_window.index(1280)
	                        hist_thresh = [960, 1280]
	                        self.over_side = True
	                        break
	                    except Exception as e:
	                        pass
	                else:
	                    if self == Left:
	                        hist_thresh = [0, 640]
	                    else:
	                        hist_thresh = [640, 1280]
	
	                j -= 90
	                i -= 90
	
	        if np.sum(x_pts) > 0:
	            self.detected = True
	        else:
	            # if the large region points search did not work, we use the results of last frame
	            x_pts = self.X
	            y_pts = self.Y
	
	        return x_pts, y_pts, self.detected, self.over_side
	
	    def find_bottom_top(self, polynomial, top_y_value=0, bot_y_value=720):
	        bottom = polynomial[0] * bot_y_value ** 2 + polynomial[1] * bot_y_value + polynomial[2]
	        top = polynomial[0] * top_y_value ** 2 + polynomial[1] * top_y_value + polynomial[2]
	        return bottom, top
	
	    def calculate_radius(self, x_pts, y_pts):
	        ym_per_pix = 30. / 720  # meters per pixel in y dimension
	        xm_per_pix = 3.7 / 700  # meteres per pixel in x dimension
	
	        fit_curve = np.polyfit(y_pts * ym_per_pix, x_pts * xm_per_pix, 2)
	        curve_radius = ((1 + (2 * fit_curve[0] * np.max(y_pts) + fit_curve[1]) ** 2) ** 1.5) / np.absolute(
	            2 * fit_curve[0])
	        return curve_radius
	
	    def sort_pts(self, x_pts, y_pts):
	        # sort all points  in ascending order
	        sorted_id = np.argsort(y_pts)
	        sorted_y_pts = y_pts[sorted_id]
	        sorted_x_pts = x_pts[sorted_id]
	        return sorted_x_pts, sorted_y_pts


## Pipeline ##

Many of the code are used to solve the problem of big road curvature, but the effect is limited.


	def pipeline(image):
	    # read distortion coefficients from pickle file
	    dist_pickle = pickle.load(
	        open("camera_cal/camera_internal_parm.pickle", "rb"))
	    mtx = dist_pickle["mtx"]
	    dist = dist_pickle["dist"]
	
	    undistorted_img = cv2.undistort(image, mtx, dist, None, mtx)
	
	    combined_binary, hls_s, luv_l, lab_b = mask_combine(undistorted_img, model=1)
	
	    # Transposed image and acquired non-zero points
	    x, y = np.nonzero(np.transpose(combined_binary))
	
	    # get x and y value of all useful points
	    if Left.detected == True:
	        left_x, left_y, Left.detected, Left.over_side = Left.quick_search(x, y)
	
	    if Right.detected == True:
	        right_x, right_y, Right.detected, Right.over_side = Right.quick_search(x, y)
	
	    if Left.detected == False:
	        left_x, left_y, Left.detected, Left.over_side = Left.slow_search(x, y, combined_binary)
	
	    if Right.detected == False:
	        right_x, right_y, Right.detected, Right.over_side = Right.slow_search(
	            x, y, combined_binary)
	
	    left_y = np.array(left_y).astype(np.float32)
	    left_x = np.array(left_x).astype(np.float32)
	    right_y = np.array(right_y).astype(np.float32)
	    right_x = np.array(right_x).astype(np.float32)
	
	    bot_y_value = 720
	
	    left_fit = np.polyfit(left_y, left_x, 2)
	    left_x_bottom, left_x_top = Left.find_bottom_top(left_fit)
	    if Left.over_side == True:
	        overside_id = left_x.index(0)
	        top_y_value = left_y[overside_id]
	        left_x_top=0
	    else:
	        top_y_value = 0
	
	    # get the intersection x value of upper(y=0) and lower(y=720) boundary
	
	
	    # storage in bottom intersection caches
	    Left.bottom.append(left_x_bottom)
	    Left.top.append(left_x_top)
	    left_x_bottom = np.mean(Left.bottom)
	    left_x_top = np.mean(Left.top)
	    Left.lastx_bottom = left_x_bottom
	    Left.lastx_top = left_x_top
	
	    # storage in points value caches
	    left_x = np.append(left_x, left_x_bottom)
	    left_y = np.append(left_y, bot_y_value)
	    left_x = np.append(left_x, left_x_top)
	    left_y = np.append(left_y, top_y_value)
	
	    # storage in top intersection caches
	
	    left_x, left_y = Left.sort_pts(left_x, left_y)
	
	    # store x and y value of all useful points in points value caches
	    Left.X = left_x
	    Left.Y = left_y
	
	    left_fit = np.polyfit(left_y, left_x, 2)
	    Left.fit_p0.append(left_fit[0])
	    Left.fit_p1.append(left_fit[1])
	    Left.fit_p2.append(left_fit[2])
	
	    # store fit line coefficient
	    left_fit = [np.mean(Left.fit_p0), np.mean(Left.fit_p1), np.mean(Left.fit_p2)]
	    left_fit_x = left_fit[0] * left_y ** 2 + left_fit[1] * left_y + left_fit[2]
	    Left.fitx = left_fit_x
	
	    # do the same step like left side
	    right_fit = np.polyfit(right_y, right_x, 2)
	    right_x_bottom, right_x_top = Right.find_bottom_top(right_fit)
	    if Right.over_side == True:
	        overside_id = right_x.index(1280)
	        top_y_value = left_y[overside_id]
	        right_x_top=1280
	    else:
	        top_y_value = 0
	
	
	
	    Right.bottom.append(right_x_bottom)
	    Right.top.append(right_x_top)
	    right_x_bottom = np.mean(Right.bottom)
	    right_x_top = np.mean(Right.top)
	    Right.lastx_bottom = right_x_bottom
	    Right.lastx_top = right_x_top
	
	    right_x = np.append(right_x, right_x_bottom)
	    right_y = np.append(right_y, 720)
	    right_x = np.append(right_x, right_x_top)
	    right_y = np.append(right_y, 0)
	
	    right_x, right_y = Right.sort_pts(right_x, right_y)
	
	    Right.X = right_x
	    Right.Y = right_y
	
	    right_fit = np.polyfit(right_y, right_x, 2)
	    Right.fit_p0.append(right_fit[0])
	    Right.fit_p1.append(right_fit[1])
	    Right.fit_p2.append(right_fit[2])
	
	    right_fit = [np.mean(Right.fit_p0), np.mean(Right.fit_p1), np.mean(Right.fit_p2)]
	    right_fit_x = right_fit[0] * right_y ** 2 + right_fit[1] * right_y + right_fit[2]
	    Right.fitx = right_fit_x
	    left_curve_radius = Left.calculate_radius(left_x, left_y)
	    right_curve_radius = Right.calculate_radius(right_x, right_y)
	
	    # average the radius of 3 frame to make it smooth
	    if Left.count % 3 == 0:
	        Left.radius = left_curve_radius
	        Right.radius = right_curve_radius
	
	    position = (right_x_bottom + left_x_bottom) / 2
	    distance_from_center = abs((640 - position) * 3.7 / 700)
	
	    # From the  bird's-eye view conversion to the normal viewing angle use re-warp-perspective
	    source = np.float32([[490, 482], [810, 482],
	                         [1250, 720], [40, 720]])
	    destination = np.float32([[0, 0], [1280, 0],
	                              [1250, 720], [40, 720]])
	
	    Minv = cv2.getPerspectiveTransform(destination, source)
	
	    warped_zero = np.zeros_like(combined_binary).astype(np.uint8)
	    color_warped = np.dstack((warped_zero, warped_zero, warped_zero))
	    pts_left = np.array([np.flipud(np.transpose(np.vstack([Left.fitx, Left.Y])))])
	    pts_right = np.array([np.transpose(np.vstack([Right.fitx, Right.Y]))])
	    pts = np.hstack((pts_left, pts_right))
	    # draw polylines in normal angle view
	    cv2.polylines(color_warped, np.int_([pts]), isClosed=False, color=(0, 0, 255), thickness=40)
	    cv2.fillPoly(color_warped, np.int_(pts), (34, 255, 34))
	    new_warped = cv2.warpPerspective(color_warped, Minv, (image.shape[1], image.shape[0]))
	    result = cv2.addWeighted(undistorted_img, 1, new_warped, 0.5, 0)
	
	    # write distance from center and radius
	    font = cv2.FONT_HERSHEY_SIMPLEX
	    if position > 640:
	        cv2.putText(result, 'Vehicle is left of center by : {:.2f}m'.format(distance_from_center), (95, 250), font,
	                    fontScale=1, color=(255, 255, 255), thickness=2)
	    else:
	        cv2.putText(result, 'Vehicle is right of center by : {:.2f}m'.format(distance_from_center), (95, 250), font,
	                    fontScale=1, color=(255, 255, 255), thickness=2)
	
	    cv2.putText(result, 'Lane curvature radius is : {}m'.format(int((Left.radius + Right.radius) / 2)), (95, 290), font,
	                fontScale=1, color=(255, 255, 255), thickness=2)
	
	    # draw different binary image masks
	
	    newCombinedImage = np.dstack((combined_binary * 255, combined_binary * 255, combined_binary * 255))
	
	    result[10:150, 100:300, :] = cv2.resize(newCombinedImage, (200, 140))
	
	    newCombinedImage2 = np.dstack((hls_s * 255, hls_s * 255, hls_s * 255))
	    result[10:150, 350:550, :] = cv2.resize(newCombinedImage2, (200, 140))
	
	    newCombinedImage3 = np.dstack((luv_l * 255, luv_l * 255, luv_l * 255))
	    result[10:150, 600:800, :] = cv2.resize(newCombinedImage3, (200, 140))
	
	    newCombinedImage4 = np.dstack((lab_b * 255, lab_b * 255, lab_b * 255))
	
	    result[10:150, 850:1050, :] = cv2.resize(newCombinedImage4, (200, 140))
	
	    cv2.putText(result, "Combined img", (95, 200), font, 1, (255, 255, 255), 2)
	    cv2.putText(result, "HLS-S ", (400, 200), font, 1, (255, 255, 255), 2)
	    cv2.putText(result, "LUV-L ", (650, 200), font, 1, (255, 255, 255), 2)
	    cv2.putText(result, "LAB-B ", (900, 200), font, 1, (255, 255, 255), 2)
	
	    Left.count += 1
	    return result

## Running Pipeline on Video ##

	if __name__ == '__main__':
	    Left = Line()
	    Right = Line()
	    vid_output = 'harder_challenge_output_final.mp4'
	    clip1 = VideoFileClip('harder_challenge_video.mp4')
	    vid_clip = clip1.fl_image(pipeline)
	    vid_clip.write_videofile(vid_output, audio=False)

## Briefly discuss any problems / issues you faced in your implementation of this project. Where will your pipeline likely fail? What could you do to make it more robust? ##

- In the process of lane line detection, the use of gradient mask with sobel operator does not simplify the problem. **Is the gradient map really useful?**
- In challenge video, the S channel of HSV has no information. **Is my threshold wrong?**
- I have tried many methods, but still cannot solve the harder-challenge perfectly.The change in light has too much effect on the white line. Trees on both sides of the road can also cause a lot of interference in the detection of the road route. When the curvature of the curve is too large, the route may disappear from view, or the left route may appear on the right. **Do you have any other solutions？**




