import pickle
import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt


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


def color_filter(image, threshold, channel, color_space):
    # color space transform
    color_img = cv2.cvtColor(image, color_space)

    # select special color channel
    color_channel = color_img[:, :, channel]
    binary_img = np.zeros_like(color_channel)
    binary_img[(color_channel >= threshold[0]) &
               (color_channel <= threshold[1])] = 1

    return binary_img


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





    return combined_binary, hls_s, luv_l, lab_b


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
