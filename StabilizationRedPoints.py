import numpy as np
import cv2



def movingAverage(curve, radius): 
  window_size = 2 * radius + 1
  # Define the filter 
  f = np.ones(window_size)/window_size 
  # Add padding to the boundaries 
  curve_pad = np.lib.pad(curve, (radius, radius), 'edge') 
  # Apply convolution 
  curve_smoothed = np.convolve(curve_pad, f, mode='same') 
  # Remove padding 
  curve_smoothed = curve_smoothed[radius:-radius]
  # return smoothed curve
  return curve_smoothed 

def smooth(trajectory): 
  smoothed_trajectory = np.copy(trajectory) 
  # Filter the x, y and angle curves
  for i in range(3):
    smoothed_trajectory[:,i] = movingAverage(trajectory[:,i], radius=SMOOTHING_RADIUS)

  return smoothed_trajectory


# The larger the more stable the video, but less reactive to sudden panning
SMOOTHING_RADIUS=150 

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

# Define the codec for output video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
 
# Set up output video
#out = cv2.VideoWriter('video_out.mp4', fourcc, fps, (w, h))
out = cv2.VideoWriter('outpy.avi',cv2.VideoWriter_fourcc('M','J','P','G'), fps, (w,h))

#red limits
lower_red = np.array([30,150,50])
upper_red = np.array([255,255,180])

# Pre-define transformation-store array
transforms = np.zeros((n_frames-1, 3), np.float32) 

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
  #cv2.imshow('res',res)

  # Convert frame to grayscale
  bgr = cv2.cvtColor(res, cv2.COLOR_HSV2BGR) 
  gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) 
  if i == 0: 
      prev_gray = gray
      gray_res = gray
  else:    
     # Detect feature points in previous frame
      prev_pts = cv2.goodFeaturesToTrack(prev_gray,
                                     maxCorners=200,
                                     qualityLevel=0.01,
                                     minDistance=20,
                                     blockSize=3)

      # Calculate optical flow (i.e. track feature points)
      curr_pts, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev_pts, None)       
      
      #update prev_gray
      gray_res = prev_gray + gray
      prev_gray = gray

      # Filter only valid points
      idx = np.where(status==1)[0]
      prev_pts = prev_pts[idx]
      curr_pts = curr_pts[idx]

      #Find transformation matrix
      #opencv < 4.0.0 -> m = cv2.estimateRigidTransform(prev_pts, curr_pts, fullAffine=False) #will only work with OpenCV-3 or less
      #m, inliners	=	cv2.estimateAffine2D(	prev_pts, curr_pts, 
      m, inliners	=	cv2.estimateAffinePartial2D(	prev_pts, curr_pts, 
                                            method = cv2.RANSAC,
                                            maxIters = 1000,   
                                            confidence = 0.9, 
                                            refineIters = 0	)
      #print (retval)
      #print (' m:') 
      #print(m)
      
      # Extract traslation
      dx = m[0,2]
      dy = m[1,2]

      # Extract rotation angle
      da = np.arctan2(m[1,0], m[0,0])
   
      # Store transformation
      transforms[i-1] = [dx,dy,da]

########################### 
#  Outside FOR 
###########################
# Compute trajectory using cumulative sum of transformations
trajectory = np.cumsum(transforms, axis=0) 
 
# Create variable to store smoothed trajectory
smoothed_trajectory = smooth(trajectory) 

# Calculate difference in smoothed_trajectory and trajectory
difference = smoothed_trajectory - trajectory
 
# Calculate newer transformation array
transforms_smooth = transforms + difference


# Reset stream to first frame 
cap.set(cv2.CAP_PROP_POS_FRAMES, 0) 


# Write n_frames-1 transformed frames
for i in range(n_frames-1):
  # Read next frame
  success, frame = cap.read() 
  if not success:
    break

  # Extract transformations from the new transformation array
  dx = transforms_smooth[i,0]
  dy = transforms_smooth[i,1]
  da = transforms_smooth[i,2]

  # Reconstruct transformation matrix accordingly to new values
  m = np.zeros((2,3), np.float32)
  m[0,0] = np.cos(da)
  m[0,1] = -np.sin(da)
  m[1,0] = np.sin(da)
  m[1,1] = np.cos(da)
  m[0,2] = dx
  m[1,2] = dy

  # Apply affine wrapping to the given frame
  frame_stabilized = cv2.warpAffine(frame, m, (w,h))
  
   #convert to HSV and separe red color
  hsv = cv2.cvtColor(frame_stabilized, cv2.COLOR_BGR2HSV)
  mask = cv2.inRange(hsv, lower_red, upper_red)
  res = cv2.bitwise_and(frame_stabilized,frame_stabilized, mask= mask)

  bgr = cv2.cvtColor(res, cv2.COLOR_HSV2BGR) 
  frame_stabilized_gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) 

  #cv2.imshow('Frame estabilizado',frame_stabilized)
  #frame_stabilized_gray = cv2.cvtColor(frame_stabilized,cv2.COLOR_BGR2GRAY) 

  #print (frame_stabilized_gray)
  #cv2.waitKey(0)

  if i == 0:
      sum_frame_stabilized = frame_stabilized_gray
  else: 
      #sum_frame_stabilized = sum_frame_stabilized + frame_stabilized_gray
      sum_frame_stabilized = cv2.addWeighted(sum_frame_stabilized,0.8,frame_stabilized_gray,0.8,0)


  cv2.imshow('Frame estabilizado',sum_frame_stabilized)
  frame2write = cv2.cvtColor(sum_frame_stabilized, cv2.COLOR_GRAY2BGR)
  out.write(frame2write)
  
 # Press Q on keyboard to  exit
  if cv2.waitKey(25) & 0xFF == ord('q'):
    break
 
out.release()
# Release video file
cap.release() 
# Closes all the frames
cv2.destroyAllWindows()
