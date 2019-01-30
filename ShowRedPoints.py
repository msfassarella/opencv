import cv2
import numpy as np

cap = cv2.VideoCapture('video.mp4')
if (cap.isOpened()== False): 
  print("Error opening video stream or file")

# Get frame count
n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) 

# Get width and height of video stream
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) 
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Get frames per second (fps)
fps = cap.get(cv2.CAP_PROP_FPS)


print ('num frames: ' + str(n_frames))
print ('size: ' + str(w) + ' x ' + str(h) )
print ('fps: ' + str(fps))

#red limits
lower_red = np.array([30,150,50])
upper_red = np.array([255,255,180])

for i in range(n_frames):
  # Read next frame
  success, frame = cap.read() 
  if not success:
    break
  #convert to HSV and separe red color
  hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
  mask = cv2.inRange(hsv, lower_red, upper_red)
  res = cv2.bitwise_and(frame,frame, mask= mask)
  
  #cv2.imshow('frame',curr)
  #cv2.imshow('mask',mask)
  cv2.imshow('res',res)

 # Press Q on keyboard to  exit
  if cv2.waitKey(25) & 0xFF == ord('q'):
    break
 
# Release video file
cap.release() 
# Closes all the frames
cv2.destroyAllWindows()
