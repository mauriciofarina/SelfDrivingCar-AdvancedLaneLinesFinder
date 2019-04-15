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

## **Perspective Transform**

In order to obtain a perspective transformation of the lane image, the source and destination coordinates were defined with:

**TODO CODE HERE**

Resulting in the values:

| Source | | Destination | |
| Source X | Source Y | Destination X | Destination Y |
| -------- | -------- | ------------- | ------------- |
| -------- | -------- | ------------- | ------------- |
| -------- | -------- | ------------- | ------------- |
| -------- | -------- | ------------- | ------------- |
| -------- | -------- | ------------- | ------------- |











