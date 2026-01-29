import keras
import tensorflow as tf
from superpoint.constants import MP_INPUT_SHAPE


# Postprocessing Layer For InterestPointDecoder to reshape outputs
class DetectorPostProcessor(keras.layers.Layer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.trainable = False
        
    def call(self, inputs):
        return tf.nn.depth_to_space(tf.nn.softmax(inputs[..., :-1], axis=-1), block_size=8)
    
    

class DescriptorPostProcessor(keras.layers.Layer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.trainable = False
        
    def call(self, inputs):
        return tf.nn.l2_normalize(
            tf.image.resize(
                tf.nn.relu(inputs), 
                size=MP_INPUT_SHAPE[:2], 
                method=tf.image.ResizeMethod.BICUBIC), 
            axis=-1
        )