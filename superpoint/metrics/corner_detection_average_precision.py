import keras
import tensorflow as tf
from superpoint.constants import (
    eta,
    cdap_dtype,
    MP_BATCH_SIZE, 
    MP_INPUT_SHAPE, 
    detection_confidences
)



class CornerDetectionAveragePrecision(keras.metrics.Metric):
    def __init__(self, dtype=None, name=None):
        super().__init__(dtype, name)
        self.batchPrecisions = self.add_weight(shape=(detection_confidences.shape[0],), initializer="zeros", dtype=cdap_dtype)
        self.batchRecalls = self.add_weight(shape=(detection_confidences.shape[0],), initializer="zeros", dtype=cdap_dtype)
        self.mAP = self.add_weight(shape=(), initializer="zeros", dtype=cdap_dtype)
        self.mLE = self.add_weight(shape=(), initializer="zeros", dtype=cdap_dtype)
        
        self.instancePrecisions = self.add_weight(shape=(detection_confidences.shape[0],), initializer="zeros", dtype=cdap_dtype)
        self.instanceRecalls = self.add_weight(shape=(detection_confidences.shape[0],), initializer="zeros", dtype=cdap_dtype)
        self.localizationError = self.add_weight(shape=(), initializer="zeros", dtype=cdap_dtype)

    
    
    @tf.function(input_signature=((
        tf.TensorSpec(shape=[None, 2], dtype=cdap_dtype),
        tf.TensorSpec(shape=MP_INPUT_SHAPE, dtype=cdap_dtype)
    ),))
    def cornerDetectionPrecision(self, instance):        
        # Removing [0, 0] coordinates from ground truth points which were added (via tf.data.Dataset.padded_batch) just to equalize the batch size
        groundTruthPoints = tf.boolean_mask(
            instance[0],
            tf.reduce_any(instance[0] != 0, axis=-1)
        )[tf.newaxis, ...]
        groundTruthPoints = tf.cast(groundTruthPoints, dtype=cdap_dtype)
        gt_points_length = tf.cast(tf.shape(groundTruthPoints)[1], cdap_dtype)
        
        index = 0
        n_detection_confidences = detection_confidences.shape[0]
        self.localizationError.assign(0.)
        for confidence in detection_confidences:
            # Getting Points where probability of pointness is greater then detection confidence
            predictedPoints = tf.cast(
                tf.sparse.from_dense(tf.greater_equal(instance[1], confidence)).indices[:, :-1], 
                cdap_dtype
            )[:, tf.newaxis, :]
            pp_points_length = tf.shape(predictedPoints)[0]
            
            # To account for special cases where FP and FN gets 0
            if tf.logical_or(
                tf.logical_and(
                    tf.equal(gt_points_length, 0),
                    tf.equal(pp_points_length, 0)
                ),
                tf.logical_and(
                    tf.equal(gt_points_length, 0),
                    tf.not_equal(pp_points_length, 0)
                )
            ):
                pass
            
            
            elif tf.logical_or(
                # When there is no TP and model predicted some (means FP)
                tf.logical_and(
                    tf.equal(gt_points_length, 0),
                    tf.not_equal(pp_points_length, 0)
                ),
                # Similarly when there are some TPs and model predicted none
                tf.logical_and(
                    tf.not_equal(gt_points_length, 0),
                    tf.equal(pp_points_length, 0)
                )
            ):
                precision = tf.constant(0, cdap_dtype)
                recall = tf.constant(0, cdap_dtype)
                
                n_detection_confidences -= 1
            else:
                # distances between predicted and ground truth points 
                distances = tf.sqrt(
                    tf.reduce_sum(
                        tf.square(
                            tf.subtract(
                                predictedPoints,
                                groundTruthPoints
                            )
                        ),
                        axis=-1
                    )
                )
                # minimum distances for each predicted point
                minimums = tf.reduce_min(distances, axis=-1)
                # Boolean Mask for distances that are less then eta  
                correctness_mask = tf.less_equal(minimums, eta)
                # Localization error for mean distance (for this confidence) that are less then eta
                # Aggregating values at all confidences to calculate mLE outside the loop
                
                
                localizationError = tf.reduce_mean(tf.boolean_mask(minimums, correctness_mask))
                #tf.print(f"Localization Error at {confidence.numpy()}: ", localizationError)
                if tf.math.is_nan(localizationError):
                    n_detection_confidences -= 1
                else:
                    self.localizationError.assign_add(localizationError)
                # Points that are at (eta or less) far away from their respective ground truth points
                correct_points = tf.boolean_mask(
                    tf.squeeze(predictedPoints, axis=1), 
                    correctness_mask
                )
                print("correct_points: ", correct_points)
                
                TP = tf.cast(tf.shape(correct_points)[0], cdap_dtype)
                FP = tf.cast(tf.shape(minimums)[0], cdap_dtype) - TP
                FN = gt_points_length - TP
                
                precision = tf.divide(TP, tf.add(TP, FP))
                recall = tf.divide(TP, tf.add(TP, FN))

            # Writing all precision values (at all confidences) for this instance 
            self.instancePrecisions[index].assign(precision)
            self.instanceRecalls[index].assign(recall)  
            index += 1
         
        # Aggregating all precisions along their respective indices
        self.batchPrecisions.assign_add(self.instancePrecisions)
        self.batchRecalls.assign_add(self.instanceRecalls)
        
        #tf.print(f"Instance Localization Error: {self.localizationError.numpy()}")
        
        return tf.divide(self.localizationError, n_detection_confidences)
    
    
    @tf.function(input_signature=(
        tf.TensorSpec(shape=[None, None, 2], dtype=cdap_dtype),
        tf.TensorSpec(shape=[None] + MP_INPUT_SHAPE, dtype=cdap_dtype)  
    ))
    def update_state(self, y_true, y_pred, sample_weight=None):
        localizationErrors = tf.map_fn(
            fn=self.cornerDetectionPrecision,
            elems=(y_true, y_pred),
            fn_output_signature=tf.TensorSpec((), cdap_dtype)
        )
        
        # Remove nans from localizationErrors
        localizationErrors = tf.boolean_mask(localizationErrors, ~tf.math.is_nan(localizationErrors))
        
        
        # Averaging precisions for each detection confidence
        self.batchPrecisions.assign(tf.divide(self.batchPrecisions, MP_BATCH_SIZE))
        self.batchRecalls.assign(tf.divide(self.batchRecalls, MP_BATCH_SIZE))
        # Averaging precisions of all detection confidences
        self.mAP.assign(tf.reduce_mean(self.batchPrecisions))
        self.mLE.assign(tf.reduce_mean(localizationErrors))
    
    
    def result(self):
        return {
            "mAP": self.mAP,
            "mLE": self.mLE,
            "precisions": self.batchPrecisions,
            "recalls": self.batchRecalls
        }
   
        
    def reset_state(self):
        self.batchPrecisions.assign(tf.zeros_like(detection_confidences))
        self.batchRecalls.assign(tf.zeros_like(detection_confidences)) 
        self.mAP.assign(0.)
        self.mLE.assign(0.)
