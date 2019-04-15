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

[image1]: ./examples/undistort_output.png "Undistorted"
[image2]: ./test_images/test1.jpg "Road Transformed"
[image3]: ./examples/binary_combo_example.jpg "Binary Example"
[image4]: ./examples/warped_straight_lines.jpg "Warp Example"
[image5]: ./examples/color_fit_lines.jpg "Fit Visual"
[image6]: ./examples/example_output.jpg "Output"
[video1]: ./project_video.mp4 "Video"

---

## **Camera Calibration**

In order to calibrate the image, a class `Camera` was created. Inside of it, fixed values of chess board corner sides were defined for the X and Y axis (9 and 6 respectively). In this class, a initialization method called `calibrateCamera` was setup in order to execute all the calibration process for the project's camera. In this method, all calibration images are loaded, converted to grayscale and processed with the `cv2.findChessboardCorners` method. After that process, the `objpoints` and `imgpoints` lists are obtained. Next, the method `cv2.calibrateCamera` and obtains the calibration parameters to be used. This method should used in the program's initialization process.

After the initialization process is done, an undistorted image can be obtained using the method `getUndistortedImage`. This method apply the calibration parameters to the distorted image using the method `cv2.undistort`.

## **Binary Image**

After obtaining the undistorted image, three steps are performed in order to obtain the final binary image:

1. First, the image is converted from the RGB color space to the HLS color space. After that, the image is devided into it's three channels. Finally, the upper and lower thresholds are applied to the saturation channel and the first binary image is obtained.

1. With the HLS channels obtained, the Sobel X is calculated for the channels saturation and light. At the end, thresholds are used in order to obtain the seccond and third binary images.

1. Finally, the resulting binary image is obtained by the result of an **OR** operation of three images from the previeous steps.

**TODO CODE HERE**

## **Perspective Transform**

In order to obtain a perspective transformation of the lane image, the source and destination coordinates were defined. After that, the perspective transoformation matrix was calculated using the method `cv2.getPerspectiveTransform`.

**TODO CODE HERE**

Resulting in the values:

| Source X | Source Y | Destination X | Destination Y |
| -------- | -------- | ------------- | ------------- |
| 100 | 100 | 100 | 100 |
| 100 | 100 | 100 | 100 |
| 100 | 100 | 100 | 100 |
| 100 | 100 | 100 | 100 |


## **Lane Lines Detection and Polynomial Fit**

As the final step for detecting the lane Lines, two aproaches were used to find lane lines and define their respective polynomials.

For the first frame, the **Sliding Window** method (`findInitialLine`) was used in order to obtain the first aproximation. In order to do so, first, it calculates the histogram for the botton half of the the image and then, calculates the middle point for the Left and Right Windows. Next, it searches for pixels inside the window and recenters the next window to the average position of the previeus one.

**TODO CODE HERE**

For the other frames, a method based on the previous polynomial (`searchAroundPoly`) is used in order to find the frame's lane lines polynomial. This method searches for all pixels close to a margin value of the previous frame polynomial. After that, it fits a new polynomial based on the average of this points.

**TODO CODE HERE**

Finally, the weighted average of the current and previous polynomial coefficients (with weights 2 and 8 respectively) is calculated, resulting in the final frame's left and right polynomials. This final process was done in order to make the a smooth transition in the polynomials found between frames.


**TODO CODE HERE**

## **Curvature and Vehicle Offset**

To calculate the lane lines curvature and the offset between the car and the center of the lane, a method called `measureCurvatureAndOffset` was created. In order to obtain the results in meters, a convertion coefficient between pixels and meters was defined as such:

|Axis|Meters per Pixels|
|----|-----------------|
| Y  | 30/720          |
| X  | 3.7/700         |


### Curvature

To calculate the curvature of the left and right lane lines, the max value of Y (Botton of the image) was choosen. After converting its value from pixels to meters, the resulting value was calculated using the equation:

**EQUATION HERE**

**TODO CODE HERE**

### Offset

Since the camera is positioned in the center of the car, the offset can be easily calculated by the difference between the middle point of the lane lines and the image center. In order to do so, the X position of the two lines was calculated using the same Y value used for the curvature (Max Y value), and the middle point between the two lines was calculated. Next, the difference between the center of the image and the middle point was obtained and finally converted to meters.

**TODO CODE HERE**



## **Visualization**

In order to display the results on the processed video, the area between the left and right polynomials was highlighted in green in the perspective transformed image. After that, the inverse of perspective transform matrix was calculated so that a highlighted version of the original image could be obtained. In the final image, the frame's results for curvature and offset were added as text for observation.




