import keras
import numpy as np
import tensorflow as tf
from superpoint.models.magicpoint import MagicPoint


class TrainingOverAllLogger(keras.callbacks.Callback):
    def __init__(self, writers, images, epoch, max_outputs=4, pred_threshold=0.5):
        super().__init__()
        self._writers = writers
        self._images = tf.convert_to_tensor(images)
        self._epoch = epoch
        self._max_outputs = max_outputs
        self._pred_threshold = pred_threshold
   

    def on_train_begin(self, epoch, logs=None):
        if self._epoch == 1:

            self.model: MagicPoint
            outputs = self.model(self._images, training=False)
            
            for block, writer in self._writers.items():
                with writer.as_default():
                    block_class = getattr(self.model.encoder, block)
                    
                    if "SE" in block:
                        # Conv2d 1
                        tf.summary.histogram(
                            "conv2d_1/Kernel",
                            block_class.conv2d_1.kernel,
                            step=0
                        )

                        tf.summary.histogram(
                            "conv2d_1/Activations",
                            outputs[block]["conv2d_1"],
                            step=0
                        )
                        
                        # BatchNorm 1
                        tf.summary.histogram(
                            "batchNorm_1/Beta",
                            block_class.batchNorm_1.beta,
                            step=0
                        )

                        tf.summary.histogram(
                            "batchNorm_1/Gamma",
                            block_class.batchNorm_1.gamma,
                            step=0
                        )

                        tf.summary.histogram(
                            "batchNorm_1/Activations",
                            outputs[block]["batchNorm_1"],
                            step=0
                        )

                        # Max Pool 1

                        # Conv2d 2
                        tf.summary.histogram(
                            "conv2d_2/Kernel",
                            block_class.conv2d_2.kernel,
                            step=0
                        )

                        tf.summary.histogram(
                            "conv2d_2/Activations",
                            outputs[block]["conv2d_2"],
                            step=0
                        )
                        
                        # Batch Norm 2
                        tf.summary.histogram(
                            "batchNorm_2/Beta",
                            block_class.batchNorm_2.beta,
                            step=0
                        )

                        tf.summary.histogram(
                            "batchNorm_2/Gamma",
                            block_class.batchNorm_2.gamma,
                            step=0
                        )
                        
                        tf.summary.histogram(
                            "batchNorm_2/Activations",
                            outputs[block]["batchNorm_2"],
                            step=0
                        )
                    
                    elif "Max" in block:
                        # MaxPool 
                        tf.summary.histogram(
                            "Activations",
                            outputs[block],
                            step=0
                        )

                writer.flush()



    def on_epoch_end(self, epoch, logs=None):
        self.model: MagicPoint
        outputs = self.model(self._images, training=False)
        
        for block, writer in self._writers.items():
            if hasattr(self.model.encoder, block):
                with writer.as_default():
                    block_class = getattr(self.model.encoder, block)

                    if "SE" in block:
                        # Conv2d 1
                        tf.summary.histogram(
                            "conv2d_1/Kernel",
                            block_class.conv2d_1.kernel,
                            step=self._epoch
                        )

                        tf.summary.histogram(
                            "conv2d_1/Activations",
                            outputs[block]["conv2d_1"],
                            step=self._epoch
                        )
                        
                        # BatchNorm 1
                        tf.summary.histogram(
                            "batchNorm_1/Beta",
                            block_class.batchNorm_1.beta,
                            step=self._epoch
                        )

                        tf.summary.histogram(
                            "batchNorm_1/Gamma",
                            block_class.batchNorm_1.gamma,
                            step=self._epoch
                        )

                        tf.summary.histogram(
                            "batchNorm_1/Activations",
                            outputs[block]["batchNorm_1"],
                            step=self._epoch
                        )

                        # Conv2d 2
                        tf.summary.histogram(
                            "conv2d_2/Kernel",
                            block_class.conv2d_2.kernel,
                            step=self._epoch
                        )

                        tf.summary.histogram(
                            "conv2d_2/Activations",
                            outputs[block]["conv2d_2"],
                            step=self._epoch
                        )
                        
                        # Batch Norm 2
                        tf.summary.histogram(
                            "batchNorm_2/Beta",
                            block_class.batchNorm_2.beta,
                            step=self._epoch
                        )

                        tf.summary.histogram(
                            "batchNorm_2/Gamma",
                            block_class.batchNorm_2.gamma,
                            step=self._epoch
                        )
                        
                        tf.summary.histogram(
                            "batchNorm_2/Activations",
                            outputs[block]["batchNorm_2"],
                            step=self._epoch
                        )
                    
                    elif "Max" in block:
                        # Max
                        tf.summary.histogram(
                            "Activations",
                            outputs[block],
                            step=self._epoch
                        )

                writer.flush()   
         