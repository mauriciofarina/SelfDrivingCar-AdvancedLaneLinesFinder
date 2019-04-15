## **Advanced Lane Line Finder**


---

**Advanced Lane Finding Project**

The steps of this project are the following:

1. Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
1. Apply a distortion correction to raw images.
1. Use color transforms, gradients, etc., to create a thresholded binary image.
1. Apply a perspective transform to rectify binary image ("birds-eye view").
1. Detect lane pixels and fit to find the lane boundary.
1. Determine the curvature of the lane and vehicle position with respect to center.
1. Warp the detected lane boundaries back onto the original image.
1. Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./report_images/distundist.png "Camera Calibration"
[image2]: ./report_images/binary.jpeg "Binary Image"
[image3]: ./report_images/birdview.png "Bird view"
[image4]: ./report_images/gotLines.png "Lane Lines"
[image5]: ./report_images/equation.png "Curvature Equation"
[image6]: ./report_images/final.png "Final Result"




---

## **Development**

### **Description**

In the `AdvancedLaneLinesFinderVideo` script the following steps are executed:

1. A `Camera` object is created and initialized by calibrating the camera.
1. A video is loaded and each frame is processed by the method `processImage`.
1. Inside `processImage` the frame is undistorted using the Camera's method `getUndistortedImage`. After that, the `pipeline` method is called and the processed image is obtained. Also, the first frame `flag` is set to false.
1. Finally, the video is saved to the output folder.

This process is very straightforward and the only observation necessary is that in the `pipeline` method the values of the previous left and right polynomial fit should be provided, as well as the distorted and undistorted images.

### **Development Files**

| File | Description |
| ------ | ------ |
| Camera.py | Camera Calibration Class |
| Pipeline.py | Pipeline Methods | 
| AdvancedLaneLinesFinderVideo.py | Runs Pipeline on Videos | 



## **Camera Calibration**

In order to calibrate the image, a class `Camera` was created. Inside of it, the fixed values of the chess board corner were defined for the X and Y axis (9 and 6 respectively). In this class, an initialization method called `calibrateCamera` was setup in order to execute all the calibration process for the project's camera. In this method, all calibration images are loaded, converted to grayscale and processed with the `cv2.findChessboardCorners` method. After that process, the `objpoints` and `imgpoints` lists were obtained. Next, the method `cv2.calibrateCamera` was executed, resulting the calibration parameters to be used. This method should used in the program's initialization process.

After the initialization process is done, an undistorted image can be obtained using the method `getUndistortedImage`. This method apply the calibration parameters to the distorted image using the method `cv2.undistort`.

```python
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
```

![alt_text][image1] 


## **Binary Image**

After obtaining the undistorted image, three steps are performed in order to obtain the final binary image:

1. First, the image is converted from the RGB color space to the HLS color space. After that, the image is divided into its three channels. Finally, the upper and lower thresholds are applied to the saturation channel and the first binary image is obtained.

1. With the HLS channels obtained, the Sobel X is calculated for the channels saturation and light. At the end, thresholds were used in order to obtain the second and third binary images.

1. Finally, the resulting binary image is obtained by the result of an **OR** operation of three images from the previous steps.

```python
# Convert Image to HLS color space
    hls = np.copy(undistortedImage)
    hls = cv2.cvtColor(hls, cv2.COLOR_RGB2HLS)
    
    # Get HLS channels
    l_channel = hls[:, :, 1]
    s_channel = hls[:, :, 2]
    
    # Get Saturation Binary
    saturationThresh = (170, 255)
    SaturationBin = np.zeros_like(s_channel)
    SaturationBin[(s_channel >= saturationThresh[0]) &
             (s_channel <= saturationThresh[1])] = 1

    # Get Sobelx of Saturation Channel
    sobelxSatThresh = (20, 100)
    sobelxSat = cv2.Sobel(s_channel, cv2.CV_64F, 1, 0)
    abs_sobelxSat = np.absolute(sobelxSat)
    scaled_sobelxSat = np.uint8(255*abs_sobelxSat/np.max(abs_sobelxSat))

    sobelxSatBin = np.zeros_like(scaled_sobelxSat)
    sobelxSatBin[(scaled_sobelxSat >= sobelxSatThresh[0]) &
             (scaled_sobelxSat <= sobelxSatThresh[1])] = 1

    # Get Sobelx of Light Channel
    sobelxLightThresh = (20, 100)
    sobelxLight = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0)
    abs_sobelxLight = np.absolute(sobelxLight)
    scaled_sobelxLight = np.uint8(255*abs_sobelxLight/np.max(abs_sobelxLight))

    sobelxLightBin = np.zeros_like(scaled_sobelxLight)
    sobelxLightBin[(scaled_sobelxLight >= sobelxLightThresh[0]) &
             (scaled_sobelxLight <= sobelxLightThresh[1])] = 1

    # Join Binary Images
    result = (SaturationBin | sobelxSatBin | sobelxLightBin)
```

![alt_text][image2]

## **Perspective Transform**

In order to obtain a perspective transformation of the lane image, the source and destination coordinates were defined. After that, the perspective transformation matrix was calculated using the method `cv2.getPerspectiveTransform`.

```python
## Perspective Transformation

    # Define Source Coordinates
    src = np.float32(
    [[(size[1] / 2) - 60, size[0] / 2 + 100],
    [((size[1] / 6) - 10), size[0]],
    [(size[1] * 5 / 6) + 60, size[0]],
    [(size[1] / 2) + 60, size[0] / 2 + 100]])


    # Calculate Destination Coordinates
    dst = np.float32(
    [[(size[1] / 4), 0],
    [(size[1] / 4), size[0]],
    [(size[1] * 3 / 4), size[0]],
    [(size[1] * 3 / 4), 0]])
    
    # Calculate Perspective Transform Matrix
    M = cv2.getPerspectiveTransform(src, dst)

    # Warp image
    warped = cv2.warpPerspective(result, M, (size[1], size[0]))
```

Resulting in the values:

| Source X | Source Y | Destination X | Destination Y |
| -------- | -------- | ------------- | ------------- |
| 580 | 460 | 320 | 0 |
| 203 | 720 | 320 | 720 |
| 1126 | 720 | 960 | 720 |
| 700 | 100 | 960 | 0 |

![alt_text][image3]


## **Lane Lines Detection and Polynomial Fit**

As the final step for detecting the lane Lines, two approaches were used to find lane lines and define their respective polynomials.

For the first frame, the **Sliding Window** method (`findInitialLine`) was used in order to obtain the first approximation. In order to do so, first, it calculates the histogram for the bottom half of the image and then, calculates the middle point for the Left and Right Windows. Next, it searches for pixels inside the window and recenter the next window to the average position of the previous one.

```python
# Find Lane Lines with Sliding Window Method
def findInitialLine(binary_warped):
    
    # Find Base Location of Left and Right Lines
    bottom_half = binary_warped[binary_warped.shape[0]//2:, :]
    histogram = np.sum(bottom_half, axis=0)

    midpoint = np.int(histogram.shape[0]//2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint



    # Choose the number of sliding windows
    nwindows = 9
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50


    # Set height of windows - based on nwindows above and image shape
    window_height = np.int(binary_warped.shape[0]//nwindows)

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
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin

        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                              (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                               (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        # If pixels found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)
    
    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    return leftx, lefty, rightx, righty
```

For the other frames, a method based on the previous polynomial (`searchAroundPoly`) is used in order to find the frame's lane lines polynomial. This method searches for all pixels close to a margin value of the previous frame polynomial. After that, it fits a new polynomial based on the average of this points.

```python
# Find Lane Lines Based on Previous Polynomial
def searchAroundPoly(binary_warped, left_fit, right_fit):
    
    # Window Margin
    margin = 100

    # Grab activated pixels
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    
    # Identify the nonzero pixels in x and y within the window
    left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + 
                    left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + 
                    left_fit[1]*nonzeroy + left_fit[2] + margin)))
    right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + 
                    right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + 
                    right_fit[1]*nonzeroy + right_fit[2] + margin)))
    
    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    
    return leftx, lefty, rightx, righty, left_fit, right_fit
```

Finally, the weighted average of the current and previous polynomial coefficients (with weights 2 and 8 respectively) is calculated, resulting in the final frame's left and right polynomials. This final process was done in order to make a smooth transition in the polynomials found between frames.


```python
## Find Polynomial

    if(flag):   # First Frame (Sliding Window)
        leftx, lefty, rightx, righty = findInitialLine(warped)
    else:   # Other Frames (Based on previous polynomial)
        leftx, lefty, rightx, righty, left_fit, right_fit = searchAroundPoly(warped, left_fit, right_fit)


    # Weighted Average of previous and current polynomial coefficients
    if left_fit is not None:
        left_fit = ((left_fit*8) + (np.polyfit(lefty, leftx, 2)*2))/10
        right_fit = ((right_fit*8) + (np.polyfit(righty, rightx, 2)*2))/10
    else:
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)
    

    # Define Left and Right Polynomial Points
    ploty = np.linspace(0, size[0]-1, size[0])
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
```

![alt_text][image4]

## **Curvature and Vehicle Offset**

To calculate the lane lines curvature and the offset between the car and the center of the lane, a method called `measureCurvatureAndOffset` was created. In order to obtain the results in meters, a convertion coefficient between pixels and meters was defined as such:

|Axis|Meters per Pixels|
|----|-----------------|
| Y  | 30/720          |
| X  | 3.7/700         |


### Curvature

To calculate the curvature of the left and right lane lines, the max value of Y (Botton of the image) was chosen. After converting its value from pixels to meters, the resulting value was calculated using the equation:

![alt_text][image5]

```python
# Calculate the Real Lane Line Curvature and The Car Offset
def measureCurvatureAndOffset(ploty, left_fit, right_fit, size):
   
    ## Curvature

    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension
    
    # Define Desired y value for Curvature
    y_eval = np.max(ploty) #(Botton of the Image)
    
    # Calculates Left and Right Curvatures
    left_curverad = ((1 + (2*left_fit[0]*y_eval*ym_per_pix + left_fit[1])**2)**1.5) / np.absolute(2*left_fit[0])
    right_curverad = ((1 + (2*right_fit[0]*y_eval*ym_per_pix + right_fit[1])**2)**1.5) / np.absolute(2*right_fit[0])
```

### Offset

Since the camera is positioned in the center of the car, the offset can be easily calculated by the difference between the middle point of the lane lines and the image center. In order to do so, the X position of the two lines was calculated using the same Y value used for the curvature (Max Y value), and the middle point between the two lines was calculated. Next, the difference between the center of the image and the middle point was obtained and finally converted to meters.

```python
## Car Offset

    # Calculate Respective x value
    xleft = left_fit[0]*y_eval**2 + left_fit[1]*y_eval + left_fit[2]
    xright = right_fit[0]*y_eval**2 + right_fit[1]*y_eval + right_fit[2]

    # Find Lane Center in Pixels
    center = ((xright - xleft)/2 ) + xleft
    # Find Car Offset in Pixels
    carOffset =  center - (size[1]/2)
    # Convert Offset from pixels to meters
    carOffset = carOffset*xm_per_pix

    return left_curverad, right_curverad, carOffset
```



## **Visualization**

In order to display the results on the processed video, the area between the left and right polynomials was highlighted in green in the perspective transformed image. After that, the inverse of perspective transform matrix was calculated so that a highlighted version of the original image could be obtained. In the final image, the frame's results for curvature and offset were added as text for observation.


```python
## Visualization

    # Create image to Draw on
    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

    # Calculate Inverse of Perspective Transform Matrix
    Minv = np.linalg.inv(M)

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (size[1], size[0]))

    # Combine the result with the original image
    result = cv2.addWeighted(distortedImage, 1, newwarp, 0.3, 0)

    # Measure Real Curvature and Offset
    left_curverad, right_curverad, carOffset = measureCurvatureAndOffset(ploty, left_fit, right_fit, size)


    # Write Values on Image
    cv2.putText(result,"Curvature Left: %.2fm" % (left_curverad),(50,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),1)
    cv2.putText(result,"Curvature Right: %.2fm" % right_curverad,(50,100),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),1)
    cv2.putText(result,"Offset: %.2fm" % carOffset,(50,150),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),1) 


    return result, left_fit, right_fit
```

![alt_text][image6]


# **Results**

## **Pipeline**

[Here](./test_videos_output/project_video.mp4) is the result of the pipeline processing for the project_video.mp4.

## **Discussion**

In this version of the pipeline, all the problems found in the challenge of project 1 were solved. However, new problems were introduced in this project. In the challenge_video, a dark line in present in the middle of the two lane lines, causing noise to the algorithm a turning it unstable. Also, the pavement is similar to some of the lane lines, causing problems on its detection as well. In the harder_challenge_video is possible that a second degree polynomial is not enough to represent the lane lines, so fitting the points to a higher degree polynomial may improve the results.
