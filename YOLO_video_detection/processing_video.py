import cv2
from darkflow.net.build import TFNet
import numpy as np
import time

option = {
    'model': 'cfg/tiny-yolo-voc-fs-1c.cfg',
    #'load': 'bin/tiny-yolo-voc.weights',
    'load': 1125,
    'threshold': 0.65,
    'gpu': 0.7
}

tfnet = TFNet(option)
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
capture = cv2.VideoCapture('matt_damon.mp4') # if you want to use video, make sure the mp4 file is in the same folder
#capture=cv2.VideoCapture(0) ##If you want use webcam use this one
out = cv2.VideoWriter('output4.avi',fourcc, 29, (854,480)) # based on the input video, change "29" the Frame rate parameter accordingly
colors = [tuple(255 * np.random.rand(3)) for i in range(5)]


while (capture.isOpened()):
    stime = time.time()
    ret, frame = capture.read()
    if ret:
        results = tfnet.return_predict(frame)
        for color, result in zip(colors, results):
            tl = (result['topleft']['x'], result['topleft']['y'])
            br = (result['bottomright']['x'], result['bottomright']['y'])
            label = result['label']
            frame = cv2.rectangle(frame, tl, br, (0,0,225),5)
            frame = cv2.putText(frame, label, tl, cv2.FONT_HERSHEY_COMPLEX, 0.8,(0, 255, 225), 2)
        cv2.imshow('frame', frame)
        out.write(frame)



        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    else:
        break
capture.release()
out.release()
cv2.destroyAllWindows()

