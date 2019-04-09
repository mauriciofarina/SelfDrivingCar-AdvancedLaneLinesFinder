import numpy as np
import cv2
import os

# Camera Calibration Class
class Camera:

    def __init__(self):
        # Define Chess Board Corners
        self.chessBoardSizeX = 9
        self.chessBoardSizeY = 6
        self.chessBoardSize = (self.chessBoardSizeX, self.chessBoardSizeY)
        self.mtx = None
        self.dist = None
        self.rvecs = None
        self.tvecs = None


    # Calibrate Camera 
    def calibrateCamera(self):

        # Calibration Images Folder
        CAL_IMG_PATH = 'camera_cal/'

        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp = np.zeros((self.chessBoardSizeY*self.chessBoardSizeX,3), np.float32)
        objp[:,:2] = np.mgrid[0:self.chessBoardSizeX, 0:self.chessBoardSizeY].T.reshape(-1,2)

        # Arrays to store object points and image points from all the images.
        objpoints = [] # 3d points in real world space
        imgpoints = [] # 2d points in image plane.

        
        for i in os.listdir(CAL_IMG_PATH):
            # Load Image
            imgOriginal = cv2.imread(CAL_IMG_PATH + i)
            img_size = (imgOriginal.shape[1], imgOriginal.shape[0])
            # Convert to Grayscale
            imgGray = cv2.cvtColor(imgOriginal, cv2.COLOR_BGR2GRAY)

            # Find Chess Board Corners
            ret, corners = cv2.findChessboardCorners(imgGray, self.chessBoardSize, None)
            if ret == True:
                objpoints.append(objp)
                imgpoints.append(corners)
                
                
        # Do camera calibration given object points and image points
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size,None,None)
        self.mtx = mtx
        self.dist = dist
        self.rvecs = rvecs
        self.tvecs = tvecs
        

    # Convert Distorted Image to Undistorted 
    def getUndistortedImage(self, image):
        return cv2.undistort(image, self.mtx, self.dist, None, self.mtx)
                
                