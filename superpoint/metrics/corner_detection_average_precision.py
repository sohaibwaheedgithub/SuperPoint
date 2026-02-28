import keras
import tensorflow as tf
from superpoint.constants import (
    eta,
    cdap_dtype, 
    MP_INPUT_SHAPE, 
    detection_confidences
)



class CornerDetectionAveragePrecision(keras.metrics.Metric):
    def __init__(self, dtype=None, name=None):
        super().__init__(dtype=dtype, name=name)
        self.batch_precisions = self.add_weight(shape=(detection_confidences.shape[0],), initializer="zeros", dtype=cdap_dtype)
        self.batch_recalls = self.add_weight(shape=(detection_confidences.shape[0],), initializer="zeros", dtype=cdap_dtype)
        self.mAP = self.add_weight(shape=(), initializer="zeros", dtype=cdap_dtype)
        self.mLE = self.add_weight(shape=(), initializer="zeros", dtype=cdap_dtype)
        
        self.instance_precisions = self.add_weight(shape=(detection_confidences.shape[0],), initializer="zeros", dtype=cdap_dtype)
        self.instance_recalls = self.add_weight(shape=(detection_confidences.shape[0],), initializer="zeros", dtype=cdap_dtype)
        self.localization_error = self.add_weight(shape=(), initializer="zeros", dtype=cdap_dtype)
        
        self.n_valid_instances = self.add_weight(shape=(), initializer="zeros", dtype=cdap_dtype)



    @tf.function(input_signature=((
        tf.TensorSpec(shape=[None, 2], dtype=cdap_dtype),
        tf.TensorSpec(shape=MP_INPUT_SHAPE, dtype=cdap_dtype)
    ),))
    def corner_detection_precision(self, instance):        
        # Remove padded [0, 0] ground-truth points added by padded_batch;
        # this also excludes images with no valid GT points, where precision and recall are undefined.
 
        ground_truth_points = tf.boolean_mask(
            instance[0],
            tf.reduce_any(instance[0] != 0, axis=-1)
        )[tf.newaxis, ...]
        ground_truth_points = tf.cast(ground_truth_points, dtype=cdap_dtype)
        gt_points_length = tf.cast(tf.shape(ground_truth_points)[1], cdap_dtype)
        
        index = tf.constant(0)
        n_localization_errors = tf.constant(0.)
     
        self.localization_error.assign(0.)

        if tf.cast(gt_points_length, tf.bool):    # To consider only those cases where precision and recall can be defined
            
            self.instance_recalls.assign(tf.zeros_like(self.instance_recalls))
            self.instance_precisions.assign(tf.zeros_like(self.instance_precisions))   

            for confidence in detection_confidences:
                # Getting Points where probability of pointness is greater then detection confidence
                predicted_points = tf.cast(
                    tf.sparse.from_dense(tf.greater_equal(instance[1], confidence)).indices[:, :-1], 
                    cdap_dtype
                )[:, tf.newaxis, :]
                pp_points_length = tf.shape(predicted_points)[0]
                

                if tf.equal(pp_points_length, 0):
                    recall = tf.constant(0, cdap_dtype)
                    precision = tf.constant(0, cdap_dtype)
                else:
                    # distances between predicted and ground truth points 
                    distances = tf.sqrt(
                        tf.reduce_sum(
                            tf.square(
                                tf.subtract(
                                    predicted_points,
                                    ground_truth_points
                                )
                            ),
                            axis=-1
                        )
                    )
                    
                    minimums = tf.reduce_min(distances, axis=-1)
                    correctness_mask = tf.less_equal(minimums, eta)
                    
                    localization_error = tf.reduce_mean(tf.boolean_mask(minimums, correctness_mask))
                    
                    if not tf.math.is_nan(localization_error):
                        self.localization_error.assign_add(localization_error)
                        n_localization_errors = n_localization_errors + 1
                    
                    # Points that are at (eta or less) far away from their respective ground truth points
                    correct_points = tf.boolean_mask(
                        tf.squeeze(predicted_points, axis=1), 
                        correctness_mask
                    )
                    
                    TP = tf.cast(tf.shape(correct_points)[0], cdap_dtype)
                    FP = tf.cast(tf.shape(minimums)[0], cdap_dtype) - TP
                    FN = gt_points_length - TP
                    
                    recall = tf.divide(TP, tf.add(TP, FN))
                    precision = tf.divide(TP, tf.add(TP, FP))

                # Writing all precision values (at all confidences) for this instance 
                self.instance_recalls[index].assign(recall)
                self.instance_precisions[index].assign(precision)  
                index = index + 1
            
            # Aggregating all precisions along their respective indices
            self.batch_recalls.assign_add(self.instance_recalls)
            self.batch_precisions.assign_add(self.instance_precisions)

            self.n_valid_instances.assign_add(1)
                    
        return tf.divide(self.localization_error, n_localization_errors)
    
    
    @tf.function(input_signature=(
        tf.TensorSpec(shape=[None, None, 2], dtype=cdap_dtype),
        tf.TensorSpec(shape=[None] + MP_INPUT_SHAPE, dtype=cdap_dtype)  
    ))
    def update_state(self, y_true, y_pred, sample_weight=None):
        localization_errors = tf.map_fn(
            fn=self.corner_detection_precision,
            elems=(y_true, y_pred),
            fn_output_signature=tf.TensorSpec((), cdap_dtype)
        )
        
        # Remove nans from localizationErrors
        localization_errors = tf.boolean_mask(localization_errors, ~tf.math.is_nan(localization_errors))
        
        
        # Averaging precisions for each detection confidence
        self.batch_precisions.assign(tf.divide(self.batch_precisions, self.n_valid_instances))
        self.batch_recalls.assign(tf.divide(self.batch_recalls, self.n_valid_instances))
        # Averaging precisions of all detection confidences
        self.mAP.assign(tf.reduce_mean(self.batch_precisions))
        self.mLE.assign(tf.reduce_mean(localization_errors))
    
    
    def result(self):
        return {
            "mAP": self.mAP,
            "mLE": self.mLE,
            "recalls": self.batch_recalls,
            "precisions": self.batch_precisions
        }
   
        
    def reset_state(self):
        self.mAP.assign(0.)
        self.mLE.assign(0.)
        self.n_valid_instances.assign(0.)
        self.batch_recalls.assign(tf.zeros_like(detection_confidences))
        self.batch_precisions.assign(tf.zeros_like(detection_confidences)) 
