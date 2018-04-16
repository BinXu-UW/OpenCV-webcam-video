For YOLO + OpenCV method
======
<p align="center">
  <img width="460" height="300" src="https://user-images.githubusercontent.com/8333665/38775463-38e9e824-4051-11e8-9e32-33bdfb890508.gif">
</p>

Requirements:  
***Tensorflow and OpenCV***  
You also need Cython: `pip install Cython`  
1. download Darkflow repository: https://github.com/thtrieu/darkflow unzip the files to a directory you like  
2. build the `darkflow` as an usable library:   
        *open a command prompt **-->**  
        *cd to the `darkflow-master` directory(e.g. D:\darkflow-master>) **-->**  
        *type `pip install -e .` or you can use `python setup.py build_ext --inplace`(Unfortunately the second command didn't work for me)  
3. download weights file or build your own weights file:  
  *Yolov2 608x608.weights file: https://pjreddie.com/media/files/yolov2.weights (You can try other weighs as well)  
  *put the weights file into `D:\darkflow-master\bin`, you need creat the **bin** folder. The folder name doesn't matter.  
  
---
## How to train your own model(not your dragon)

1. Find `tiny-yolo-voc.cfg` in your directory i.e. `D:\darkflow-master`. Copy it and rename it to `tiny-yolo-voc-1c.cfg` (1c means you only have 1 class to detect) or anyname you like.  

2. Open the `tiny-yolo-voc-1c.cfg`, chage the last [region] layer `class=20` to the number of classes you are going to train for. In our case, we change it to `class=1`.  

3. Go to the [convolutional] layer (the second to last layer) to num * (classes + 5). In our case, num is 5 and classes are 1 so 5 * (1 + 5) = 30 therefore filters are set to 30. `filters=30`.  

4. Go to the `D:\darkflow-master` find `labels.txt`, change the content to the name of your classes' name. 
In our case, we write `Matt_Damon` and save it.  

5. Go to https://pjreddie.com/darknet/yolo/ and download Tiny YOLO weights https://pjreddie.com/media/files/yolov2-tiny.weights
and put it into the bin folder.

6. * In the `D:\darkflow-master` directory open cmd.  
   * type`python flow --model cfg/tiny-yolo-voc-1c.cfg --load bin/yolov2-tiny.weights --train --annotation my_models/annotations --dataset my_models/images --gpu 0.7 --epoch 400`.  
   * If gpu usage sets greater then`--gpu 0.7` you may get an error like this:`F tensorflow/core/kernels/conv_ops.cc:605] Check failed: stream->parent()->GetConvolveAlgorithms(&algorithms)`  
   * You may also get this error: `AssertionError: expect 44948596 bytes, found 44948600`.You need go to`D:\darkflow-master\darkflow\utils`
   modify **loader.py**. Change the line 121 from self.offset = 16 to self.offset = 20

7. Wait until loss become less than 1.0 or near 1.0. I only get to 1.2 and it can not go down any lower. Then type`Ctrl-c` kill it.  

8.
