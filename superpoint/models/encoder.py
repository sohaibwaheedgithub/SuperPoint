import keras


# Convolutional Block For Shared Encoder
class SEConvBlock(keras.layers.Layer):
    def __init__(self, filters, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._filters = filters
        self.conv2d_1 = keras.layers.Conv2D(
            filters=filters,
            kernel_size=3, 
            padding="same", 
            activation="relu", 
            kernel_initializer="he_normal"
        )  
        self.batchNorm_1 = keras.layers.BatchNormalization()
        self.conv2d_2 = keras.layers.Conv2D(
            filters=filters, 
            kernel_size=3, 
            padding="same", 
            activation="relu", 
            kernel_initializer="he_normal"
        )
        self.batchNorm_2 = keras.layers.BatchNormalization()
        
        
    def call(self, inputs, training=None):
        return self.batchNorm_2(
            self.conv2d_2(
                self.batchNorm_1(
                    self.conv2d_1(inputs),
                    training=training
                )
            ),
            training=training
        )

    def compute_output_shape(self, input_shape):
        if input_shape is None:
            return None
        batch, height, width, _channels = input_shape
        return (batch, height, width, self._filters)
        
# Shared Encoder
class SharedEncoder(keras.layers.Layer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.maxPool = keras.layers.MaxPool2D(pool_size=2, strides=2)
        self.SEConvBlock_1 = SEConvBlock(64)
        self.SEConvBlock_2 = SEConvBlock(64)
        self.SEConvBlock_3 = SEConvBlock(128)
        self.SEConvBlock_4 = SEConvBlock(128)
        
        
    
    def call(self, inputs, training=None):
        return self.SEConvBlock_4(
            self.maxPool(
                self.SEConvBlock_3(
                    self.maxPool(
                        self.SEConvBlock_2(
                            self.maxPool(
                                self.SEConvBlock_1(inputs, training=training)
                            ),
                            training=training
                        )
                    ),
                    training=training
                )
            ),
            training=training
        )


    def compute_output_shape(self, input_shape):
        shape = self.SEConvBlock_1.compute_output_shape(input_shape)
        shape = self.maxPool.compute_output_shape(shape)
        shape = self.SEConvBlock_2.compute_output_shape(shape)
        shape = self.maxPool.compute_output_shape(shape)
        shape = self.SEConvBlock_3.compute_output_shape(shape)
        shape = self.maxPool.compute_output_shape(shape)
        shape = self.SEConvBlock_4.compute_output_shape(shape)
        return shape
    


if __name__ == "__main__":
    import tensorflow as tf
    encoder = SharedEncoder()
    x = tf.random.normal((1, 240, 320, 1))  # batch, height, width, channels
    y = encoder(x, training=False)

    print("Output shape:", y.shape)
