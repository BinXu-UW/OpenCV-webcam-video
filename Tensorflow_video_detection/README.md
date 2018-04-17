#This is for Tensorflow method
---


The origin files come from Tensorflow models:  
https://github.com/tensorflow/models/tree/master/research/object_detection  
You need download this and put "webcam_detection.py" into this directory  
You may get error like this:  
`from object_detection.protos import string_int_label_map_pb2 ModuleNotFoundError: No module named 'object_detection'`  
to fix this go to the utils/label_map_util.py change `from object_detection.protos import string_int_label_map_pb2` to `from protos import string_int_label_map_pb2`  
or an error like this:  
`from object_detection.core import standard_fields as fields ModuleNotFoundError: No module named 'object_detection'`
to fix this go to the /utils/visualiztion_utils.py change `from object_detection.core import standard_fields as fields` to `from core import standard_fields as fields`  
This is due to upgraded version, the directory of certain files got changed  
I will change the "webcam_detection.py" later


---
***You may need install Tensorflow***

For CPU  
`pip install tensorflow`  
For GPU  
`pip install tensorflow-gpu`

---
***for installing opencv***  
this is for `import cv2`  
`pip install opencv-python`

---

***Protobuf Compilation***  
download protobuf 3.4.0 https://github.com/google/protobuf/releases/tag/v3.4.0  
I tried version 3.5.0 of protobuf on Windows but it didn't work for me.  
unzip the file, find "bin" open it, copy the "protoc.exe" and put it into the "tensorflow/models/tree/master/research" directory  
open command prompt-->go to the directory where you put "tensorflow/models/.../research"-->  
type: `protoc object_detection/protos/*.proto --python_out=.` this will compile all the .proto files in the protos folder  

---
Then you can run the "webcam_detection.py"  

---
If you need other Tensorflow detection models, here is the link:  
https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md
