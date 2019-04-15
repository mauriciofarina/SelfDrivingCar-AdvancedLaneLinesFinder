import numpy as np
import cv2


# Pipeline to Find Lane Lines
def pipeline(distortedImage, undistortedImage, flag, left_fit, right_fit):

    # Load Image
    distortedImage = distortedImage
    undistortedImage = undistortedImage
    size = (distortedImage.shape[0], distortedImage.shape[1])
    

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

