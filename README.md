# OpenCV-webcam-video
---

This project is mean to use OpenCV, Tensorflow and other packages to detect certain object in a video or an real-time webcam recording. 
I'm using **Python 3.6, Windows 10**   

The following packages will be used:  
 * OpenCV: `pip install opencv-python`  
 * Tensorflow: For CPU `pip install tensorflow`; For GPU `pip install tensorflow-gpu`  
 * YOLO(darkflow): https://github.com/thtrieu/darkflow  
 * numpy `pip install numpy`  
 * json  
 * BeautifulSoup `pip install beautifulsoup4`  

**Tensorflow_video_detection**: This is an experiment on using the basic Tensorflow packages to 
detect certain object in video with pre-trained model.  

**YOLO_video_detection**: This is our main method. It is faster and more accurate than baisc Tensorflow
detection. We will use this method to detect certain person in a video. we will train our model to recogniz
the actor "matt damon". Because as an actor there are huge amount of pics online, so it is easy to crawl them
down and make it as a customized training model. I will train myself as a model as well later on.  

<p align="center">
  <img width="460" height="300" src="https://user-images.githubusercontent.com/8333665/38754168-c6f6c7e6-3f2e-11e8-8c57-a0c5e28303d5.png">
</p>
