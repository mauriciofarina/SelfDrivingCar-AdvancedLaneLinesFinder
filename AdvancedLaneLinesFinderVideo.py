from moviepy.editor import VideoFileClip
from Pipeline import pipeline 
from Camera import Camera

# Define Video Input/Output Path and File
VIDEO_INPUT_PATH = 'test_videos/'
VIDEO = 'project_video.mp4'
VIDEO_OUTPUT_PATH = 'test_videos_output/'

# Calibrate Camera
cam = Camera()
cam.calibrateCamera()

# Feedback Variables
flag = True # Is First Frame
left_fit = None # Left Polynomial Coefficients
right_fit = None # Right Polynomial Coefficients

# Run Pipeline on Frame
def processImage(image):
    global flag
    global left_fit
    global right_fit

    # Get Undistorted Frame Image
    imageUndist = cam.getUndistortedImage(image)
    # Run Pipeline
    result, left_fit, right_fit = pipeline(image, imageUndist, flag, left_fit, right_fit)

    flag = False

    return result

# Output Video Path
whiteOutput = VIDEO_OUTPUT_PATH + VIDEO
# Load Input Video
clip1 = VideoFileClip(VIDEO_INPUT_PATH+ VIDEO)
# Process Video
whiteClip = clip1.fl_image(processImage)
# Save Video
whiteClip.write_videofile(whiteOutput)
    






