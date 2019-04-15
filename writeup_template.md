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

[image1]: ./report_images/distImg.png "Distorted"
[image2]: ./report_images/distImg.png "Undistorted"
[image3]: ./report_images/satbin.png "Saturation Channel Binary"
[image4]: ./report_images/sobelsatbin.png "Saturation Channel Sobel X Binary"
[image5]: ./report_images/sobellightbin.png "Light Channel Sobel X Binary"
[image6]: ./report_images/resultbin.png "Result Binary"
[image7]: ./report_images/birdview.png "Bird view"
[image8]: ./report_images/gotLines.png "Lane Lines"
[image9]: ./report_images/equation.png "Curvature Equation"
[image10]: ./report_images/final.png "Final Result"


[video1]: .test_videos_output/project_video.mp4 "Video"

---

## **Development**

### **Description**

In the `AdvancedLaneLinesFinderVideo` script the following steps are executed:

1. A `Camera` object is created and initialized by calibrating the camera.
1. A video is loaded and each frame is processed by the method `processImage`.
1. Inside `processImage` the frame is undistorted using the Camera's method `getUndistortedImage`. After that, the `pipeline` method is called and the processed image is obtained. Also, the is first frame `flag` is set to false.
1. Finally, the video is saved to the output folder.

This process is very straightforward and the only observation necessary is that in the `pipeline` method the values of the previous left and right polynomial fit should be provided, as well as the distorted and undistorted images.

### **Development Files**

| File | Description |
| ------ | ------ |
| Camera.py | Camera Calibration Class |
| Pipeline.py | Pipeline Methods | 
| AdvancedLaneLinesFinderVideo.py | Runs Pipeline on Videos | 



## **Camera Calibration**

In order to calibrate the image, a class `Camera` was created. Inside of it, the fixed values of the chess board corner were defined for the X and Y axis (9 and 6 respectively). In this class, a initialization method called `calibrateCamera` was setup in order to execute all the calibration process for the project's camera. In this method, all calibration images are loaded, converted to grayscale and processed with the `cv2.findChessboardCorners` method. After that process, the `objpoints` and `imgpoints` lists were obtained. Next, the method `cv2.calibrateCamera` was executed, resulting the calibration parameters to be used. This method should used in the program's initialization process.

After the initialization process is done, an undistorted image can be obtained using the method `getUndistortedImage`. This method apply the calibration parameters to the distorted image using the method `cv2.undistort`.

```python

```

## **Binary Image**

After obtaining the undistorted image, three steps are performed in order to obtain the final binary image:

1. First, the image is converted from the RGB color space to the HLS color space. After that, the image is divided into its three channels. Finally, the upper and lower thresholds are applied to the saturation channel and the first binary image is obtained.

1. With the HLS channels obtained, the Sobel X is calculated for the channels saturation and light. At the end, thresholds were used in order to obtain the seccond and third binary images.

1. Finally, the resulting binary image is obtained by the result of an **OR** operation of three images from the previeous steps.

```python

```

## **Perspective Transform**

In order to obtain a perspective transformation of the lane image, the source and destination coordinates were defined. After that, the perspective transoformation matrix was calculated using the method `cv2.getPerspectiveTransform`.

```python

```

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

```python

```

For the other frames, a method based on the previous polynomial (`searchAroundPoly`) is used in order to find the frame's lane lines polynomial. This method searches for all pixels close to a margin value of the previous frame polynomial. After that, it fits a new polynomial based on the average of this points.

```python

```

Finally, the weighted average of the current and previous polynomial coefficients (with weights 2 and 8 respectively) is calculated, resulting in the final frame's left and right polynomials. This final process was done in order to make the a smooth transition in the polynomials found between frames.


```python

```

## **Curvature and Vehicle Offset**

To calculate the lane lines curvature and the offset between the car and the center of the lane, a method called `measureCurvatureAndOffset` was created. In order to obtain the results in meters, a convertion coefficient between pixels and meters was defined as such:

|Axis|Meters per Pixels|
|----|-----------------|
| Y  | 30/720          |
| X  | 3.7/700         |


### Curvature

To calculate the curvature of the left and right lane lines, the max value of Y (Botton of the image) was choosen. After converting its value from pixels to meters, the resulting value was calculated using the equation:

**EQUATION HERE**

```python

```

### Offset

Since the camera is positioned in the center of the car, the offset can be easily calculated by the difference between the middle point of the lane lines and the image center. In order to do so, the X position of the two lines was calculated using the same Y value used for the curvature (Max Y value), and the middle point between the two lines was calculated. Next, the difference between the center of the image and the middle point was obtained and finally converted to meters.

```python

```



## **Visualization**

In order to display the results on the processed video, the area between the left and right polynomials was highlighted in green in the perspective transformed image. After that, the inverse of perspective transform matrix was calculated so that a highlighted version of the original image could be obtained. In the final image, the frame's results for curvature and offset were added as text for observation.


```python

```

