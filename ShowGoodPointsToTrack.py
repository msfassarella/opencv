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

# Define the codec for output video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

# Set up output video
out = cv2.VideoWriter('video_out.mp4', fourcc, fps, (w, h))


# Read first frame
ret, prev = cap.read() 

if ret == True:
   # Convert frame to grayscale
   prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY) 

   # Pre-define transformation-store array
   transforms = np.zeros((n_frames-1, 3), np.float32) 

 # Detect feature points in previous frame
   prev_pts = cv2.goodFeaturesToTrack(prev_gray,
                                     maxCorners=20,
                                     qualityLevel=0.01,
                                     minDistance=15,
                                     blockSize=3)
   
   print('dim: '  + str(prev_pts.ndim))
   print('shape: '  + str(prev_pts.shape))
   print('size: ' + str(prev_pts.size))
   print('ponto: ' + str(prev_pts[0,0]))
   print('x: ' + str(prev_pts[0,0,0]))
   print('y: ' + str(prev_pts[0,0,1]))
   x = int(prev_pts[0,0,0])
   y = int(prev_pts[0,0,1])
   print('ponto: ' + str((10,10)))

   
   for cont in range(prev_pts.shape[0]): 
      x = int(prev_pts[cont,0,0])
      y = int(prev_pts[cont,0,1])
      cv2.drawMarker(prev, (x,y),(0,0,255))

 
    # Display the resulting frame
   cv2.imshow('Frame',prev)
    #out.write(frame) 
    # Press Q on keyboard to  exit
   cv2.waitKey(0)
 
 
# rastreando os pontos no proximo frame
# Convert to grayscale
sucess,curr = cap.read()
curr_gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY) 
# Calculate optical flow (i.e. track feature points)
curr_pts, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts, None)  
# When everything done, release the video capture object
 
# Sanity check
assert prev_pts.shape == curr_pts.shape 

  # Filter only valid points
idx = np.where(status==1)[0]
prev_pts = prev_pts[idx]
curr_pts = curr_pts[idx]

print('shape: '  + str(prev_pts.shape))

for cont in range(curr_pts.shape[0]): 
   x = int(curr_pts[cont,0,0])
   y = int(curr_pts[cont,0,1])
   cv2.drawMarker(prev, (x,y),(0,255,0))

cv2.imshow('Frame',prev)
cv2.waitKey(0)

cap.release()
 
# Closes all the frames
cv2.destroyAllWindows()
out.release()