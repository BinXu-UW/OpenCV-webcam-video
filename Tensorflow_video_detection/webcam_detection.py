# In[ ]:
import numpy as np
import os
import six.moves.urllib as urllib

import tarfile
import tensorflow as tf

from utils import label_map_util
from utils import visualization_utils as vis_util



# In[ ]:


# What model to download.
MODEL_NAME = 'ssd_mobilenet_v1_coco_2017_11_17'
MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')

NUM_CLASSES = 90


# ## Download Model

# In[ ]:


opener = urllib.request.URLopener()
opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
tar_file = tarfile.open(MODEL_FILE)
for file in tar_file.getmembers():
  file_name = os.path.basename(file.name)
  if 'frozen_inference_graph.pb' in file_name:
    tar_file.extract(file, os.getcwd())


# ## Load a (frozen) Tensorflow model into memory.

# In[ ]:


detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')


# ## Loading label map
# Label maps map indices to category names, so that when our convolution network predicts `5`, we know that this corresponds to `airplane`.  Here we use internal utility functions, but anything that returns a dictionary mapping integers to appropriate string labels would be fine

# In[ ]:


label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)


# ## Detection

# In[ ]:
import cv2
cap=cv2.VideoCapture("video.mp4") 
# if you want to use video, make sure the mp4 file is in the same folder

#cap=cv2.VideoCapture(0) ##If you want use webcam use this one

with detection_graph.as_default():
    with tf.Session(graph=detection_graph) as sess:
        ret =True
        while (ret):
            ret,image_np=cap.read()
            image_np_expanded = np.expand_dims(image_np, axis=0)
            image_tensor =detection_graph.get_tensor_by_name('image_tensor:0')
            boxes=detection_graph.get_tensor_by_name('detection_boxes:0')
            scores=detection_graph.get_tensor_by_name('detection_scores:0')
            classes=detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections=detection_graph.get_tensor_by_name('num_detections:0')
            
            (boxes,scores,classes,num_detections)= sess.run(
                    [boxes,scores,classes,num_detections],
                    feed_dict={image_tensor:image_np_expanded})
            
            vis_util.visualize_boxes_and_labels_on_image_array(
                    image_np,
                    np.squeeze(boxes),
                    np.squeeze(classes).astype(np.int32),
                    np.squeeze(scores),
                    category_index,
                    use_normalized_coordinates=True,
                    line_thickness=8)
            cv2.imshow('image',cv2.resize(image_np,(800,600)))
            if cv2.waitKey(25)&0xFF==ord('q'):
                break
                
        cap.release()
        cv2.destroyAllWindows()
        
            
 
  

