import keras


# Decoder Head For Both Detector And Descriptor
class Decoder(keras.layers.Layer):
    def __init__(self, filters, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv2d = keras.layers.Conv2D(256, 3, padding="same", activation="relu", kernel_initializer="he_normal")
        self.batchNorm = keras.layers.BatchNormalization()
        self.bottleNeckLayer = keras.layers.Conv2D(filters, 1, padding="same", kernel_initializer="he_normal")
        
    def call(self, input, training=None):
        return self.bottleNeckLayer(
            self.batchNorm(
                self.conv2d(input),
                training=training
            )
        )
