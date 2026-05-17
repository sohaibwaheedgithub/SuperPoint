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
            kernel_initializer="he_normal",
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
        # return self.batchNorm_2(
        #     self.conv2d_2(
        #         self.batchNorm_1(
        #             self.conv2d_1(inputs),
        #             training=training
        #         )
        #     ),
        #     training=training
        # )
        
        conv2d_1 = self.conv2d_1(inputs)
        batchNorm_1 = self.batchNorm_1(conv2d_1, training=training)
        conv2d_2 = self.conv2d_2(batchNorm_1)
        batchNorm_2 = self.batchNorm_2(conv2d_2, training=training)

        return {
            "conv2d_1": conv2d_1,
            "batchNorm_1": batchNorm_1,
            "conv2d_2": conv2d_2,
            "batchNorm_2": batchNorm_2
        }
    


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
        # return self.SEConvBlock_4(
        #     self.maxPool(
        #         self.SEConvBlock_3(
        #             self.maxPool(
        #                 self.SEConvBlock_2(
        #                     self.maxPool(
        #                         self.SEConvBlock_1(inputs, training=training)
        #                     ),
        #                     training=training
        #                 )
        #             ),
        #             training=training
        #         )
        #     ),
        #     training=training
        # )

        SEConvBlock_1 = self.SEConvBlock_1(inputs, training=training)
        maxPool_1 = self.maxPool(SEConvBlock_1["batchNorm_2"])
        SEConvBlock_2 = self.SEConvBlock_2(maxPool_1, training=training)
        maxPool_2 = self.maxPool(SEConvBlock_2["batchNorm_2"])
        SEConvBlock_3 = self.SEConvBlock_3(maxPool_2, training=training)
        maxPool_3 = self.maxPool(SEConvBlock_3["batchNorm_2"])
        SEConvBlock_4 = self.SEConvBlock_4(maxPool_3, training=training)

        return {
            "SEConvBlock_1": SEConvBlock_1,
            "maxPool_1": maxPool_1,
            "SEConvBlock_2": SEConvBlock_2,
            "maxPool_2": maxPool_2,
            "SEConvBlock_3": SEConvBlock_3,
            "maxPool_3": maxPool_3,
            "SEConvBlock_4": SEConvBlock_4,
        }



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
