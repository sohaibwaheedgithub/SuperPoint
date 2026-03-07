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

    def compute_output_shape(self, input_shape):
        if input_shape is None:
            return None
        batch, height, width, channels = input_shape
        if channels is None:
            out_channels = None
        else:
            out_channels = (channels - 1) // (8 * 8)
        if height is None or width is None:
            return (batch, None, None, out_channels)
        return (batch, height * 8, width * 8, out_channels)
    
    

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

    def compute_output_shape(self, input_shape):
        if input_shape is None:
            return None
        batch, _height, _width, channels = input_shape
        return (batch, MP_INPUT_SHAPE[0], MP_INPUT_SHAPE[1], channels)
