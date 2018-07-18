

# Import packages
import os
import cv2
import numpy as np
import tensorflow as tf
import sys
from grabscreen import grab_screen

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")

# Import utilites
from utils import label_map_util
from utils import visualization_utils as vis_util
import pyautogui
import pygame
import ctypes
import PIL
import win32api
import win32con
import win32com
import win32com.client
import win32gui
import win32ui
import time
import mss


def moveMouseRel(xOffset, yOffset):  # X ve Y' ye Bağlı Hareket Fonksyonu
    ctypes.windll.user32.mouse_event(0x0001, xOffset, yOffset, 0, 0)


# Name of the directory containing the object detection module we're using
MODEL_NAME = 'inference_graph'

# Grab path to current working directory
CWD_PATH = os.getcwd()

# Path to frozen detection graph .pb file, which contains the model that is used
# for object detection.
PATH_TO_CKPT = os.path.join(CWD_PATH, MODEL_NAME, 'frozen_inference_graph.pb')

# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH, 'training', 'labelmap.pbtxt')

# Number of classes the object detector can identify
NUM_CLASSES = 3

## Load the label map.
# Label maps map indices to category names, so that when our convolution
# network predicts `5`, we know that this corresponds to `king`.
# Here we use internal utility functions, but anything that returns a
# dictionary mapping integers to appropriate string labels would be fine
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                            use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# Load the Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

with detection_graph.as_default():
    with tf.Session(graph=detection_graph) as sess:
        while True:
            # screen = cv2.resize(grab_screen(region=(0,40,1280,745)), (WIDTH,HEIGHT))
            screen = cv2.resize(grab_screen(region=(0, 40, 1920, 1080)), (800, 450))
            image_np = cv2.cvtColor(screen, cv2.COLOR_BGR2RGB)
            # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
            image_np_expanded = np.expand_dims(image_np, axis=0)
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            # Each box represents a part of the image where a particular object was detected.
            boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            # Each score represent how level of confidence for each of the objects.
            # Score is shown on the result image, together with the class label.
            scores = detection_graph.get_tensor_by_name('detection_scores:0')
            classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')
            # Actual detection.
            (boxes, scores, classes, num_detections) = sess.run(
                [boxes, scores, classes, num_detections],
                feed_dict={image_tensor: image_np_expanded})
            # Visualization of the results of a detection.
            vis_util.visualize_boxes_and_labels_on_image_array(
                image_np,
                np.squeeze(boxes),
                np.squeeze(classes).astype(np.int32),
                np.squeeze(scores),
                category_index,
                use_normalized_coordinates=True,
                line_thickness=8)
            for i, b in enumerate(boxes[0]):
                #                 car                    bus                  truck
                if classes[0][i] == 1:
                    if scores[0][i] >= 0.5:
                        mid_x = (boxes[0][i][1] + boxes[0][i][3]) / 2
                        mid_y = (boxes[0][i][0] + boxes[0][i][2]) / 2
                        # apx_distance = round(((1 - (boxes[0][i][3] - boxes[0][i][1])) ** 4), 1)
                        x = mid_x * 1920
                        y = mid_y * 1080
                        print("mid_x: ", x, "mid_y: ", y)

                        # cv2.putText(image_np, '{}'.format(apx_distance), (int(mid_x*800),int(mid_y*450)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
                        if x > 960 and y < 540:  # 1. Bölge
                            x_new = (x - 960) + 15
                            y_new = -(540 - y) + 15
                            print("x_new: ", x_new, "y_new: ", y_new)
                            print("1.BÖLGE")
                        elif x < 960 and y < 540:  # 2. Bölge
                            x_new = -(960 - x) + 15
                            y_new = -(540 - y) + 15
                            print("2.BÖLGE")
                        elif x < 960 and y > 540:  # 3.Bölge
                            x_new = -(960 - x) + 15
                            y_new = y - 540 + 15
                            print("3.bölge")
                            print("x_new: ", x_new, "y_new: ", y_new)
                        else:
                            x_new = x - 960 + 15
                            y_new = y - 540 + 15
                            print("4.bölge")
                            print("x_new: ", x_new, "y_new: ", y_new)

                        moveMouseRel(int(x_new), int(y_new))
                        moveMouseRel(0,0)
                        pyautogui.click(int(x_new), int(y_new))
                        moveMouseRel(int(-x_new), int(-y_new))

                cv2.imshow('window', image_np)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break
