#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf


class KeyPointClassifier(object):
    def __init__(
        self,
        model_path='model/keypoint_classifier/keypoint_classifier.tflite',
        num_threads=1,
    ):
        self.interpreter = tf.lite.Interpreter(model_path=model_path,
                                               num_threads=num_threads)

        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    # Classes : 
    '''
    0 - Open          :        Panning  
    1 - Close         :        (No function)
    2 - Pointer       :        Draw
    3 - OK            :        Set start point 
    4 - Thumb Up      :        Change mode up
    5 - Thumb Down    :        Change mode down
    6 - Alt_Ok        :        Set end point
    7 - Yo            :        Eraser
    '''

    def __call__(
        self,
        landmark_list,
        class_thresholds={2: 0.4, 3: 0.5, 6: 0.8, 0: 0.7, 7: 0.7}  
    ):
        input_details_tensor_index = self.input_details[0]['index']
        self.interpreter.set_tensor(
            input_details_tensor_index,
            np.array([landmark_list], dtype=np.float32))
        self.interpreter.invoke()
        output_details_tensor_index = self.output_details[0]['index']
        result = self.interpreter.get_tensor(output_details_tensor_index)
        
        # Get the probabilities
        probabilities = np.squeeze(result)
        
        # Find the class with the highest probability
        max_class = np.argmax(probabilities)
        max_prob = probabilities[max_class]
        
        # Apply custom thresholds for specific classes
        if max_class in class_thresholds and max_prob < class_thresholds[max_class]:
            # If the probability doesn't meet the threshold, find the next highest class
            # or return a "unknown" class (e.g., -1)
            probabilities[max_class] = 0  # Zero out the insufficient probability
            second_max_class = np.argmax(probabilities)
            return second_max_class if probabilities[second_max_class] > 0 else -1
        
        return max_class
