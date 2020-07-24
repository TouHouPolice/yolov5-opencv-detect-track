import cv2
import numpy as np

video_path="inference/videos/MOT16-06-raw.webm"
img_path="inference/images/bus.jpg"

videocap=cv2.VideoCapture(video_path)
#print (videocap.isOpened())
img=cv2.imread(img_path)
#print(img)
cv2.imshow("image",img)
cv2.waitKey(0)
ret=1
counter=0
ret,frame=videocap.read()
img=np.array(img)

while ret:
    print(str(counter))
    counter+=1
    ret,frame=videocap.read()
    cv2.imshow("video",frame)
    cv2.waitKey(1)

videocap.release()

