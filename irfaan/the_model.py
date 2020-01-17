import tensorflow as tf
import numpy as np
import cv2

class detector(object):

    def __init__(self,PATH_TO_MODEL):
        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            # Works up to here.
            with tf.gfile.GFile(PATH_TO_MODEL, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
            self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
            self.d_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0') 
            self.d_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
            self.d_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
            self.num_d = self.detection_graph.get_tensor_by_name('num_detections:0')
        self.sess = tf.Session(graph=self.detection_graph)

    def get_classification(self, img):
        #Bounding Box Detection.
        ## The model we trained needs to be fed the image in this form
        #  img_np = img[:,:,[2,1,0]]
        img = img[:,:,[2,1,0]]
        rows = img.shape[0]
        cols = img.shape[1]
        with self.detection_graph.as_default():
            # Expand dimension since the model expects image to have shape [1, None, None, 1].
            img_expanded = np.expand_dims(img, axis=0)  
            (boxes, scores, classes, num) = self.sess.run(
                [self.d_boxes, self.d_scores, self.d_classes, self.num_d],
                feed_dict={self.image_tensor: img_expanded})
            boxes = boxes.reshape((boxes.shape[1],boxes.shape[2]))
            scores = scores.reshape((scores.shape[1],))
            ind = tf.image.non_max_suppression(boxes, scores, 200, 0.12)
            indices = self.sess.run(ind)
            boxes = np.expand_dims(boxes[indices] , axis = 0)
            scores = np.expand_dims(scores[indices], axis = 0)

        # print(boxes, scores, classes[:,indices], len(indices))

        out = boxes, scores, classes[:,indices], len(indices)
        num_detections = int(out[3])
                
        scores = []
        classes = []
        boxes = []
        true_detections = 0
        for i in range(num_detections):
            classId = int(out[2][0][i])
            score = float(out[1][0][i])
            bbox = [float(v) for v in out[0][0][i]]
            
            if score > 0.5:
                true_detections += 1
                # print(score , end = '\r')
                x = int(bbox[1] * cols)
                y = int(bbox[0] * rows)
                right = int(bbox[3] * cols)
                bottom = int(bbox[2] * rows)
                bbox=(x,y,right-x,bottom-y)
                # ybboxes.append(bbox)
                tl = (x,y)
                br = (right,bottom)
                if classId == 1: #is it person????
                    classes.append('1')
                else:
                    classes.append('0')
                scores.append(score)

                pt1 = dict(x = x, y = y)
                pt2 = dict(x = right, y = bottom)
                boxes.append({'topleft' : pt1, 'bottomright' : pt2})
                # cv2.rectangle(img,tl, br, (0,0,255), 2, 1)
                # cv2.putText(img,
                #     str(classId) +
                #     " [" + str(round(score * 100, 2)) + "]",
                #     (tl[0], br[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                #     [0, 255, 0], 2)
        # cv2.imshow("frame",img)
        cv2.waitKey(1)
        print( boxes, [scores], classes, true_detections)

        return boxes, [scores], classes, true_detections
