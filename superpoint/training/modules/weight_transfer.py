import keras
from superpoint.models.components.encoder import SharedEncoder
from superpoint.models.components.post_processor import DetectorPostProcessor


def _transfer_weights(magicpoint: keras.Model):
    """Transfer weights from MagicPoint to SuperPoint for the shared components"""
    # Extract weights from MagicPoint
    for layer in magicpoint.layers:
        if isinstance(layer, SharedEncoder) or isinstance(layer, DetectorPostProcessor) or isinstance(layer, IPDPostProcessor):
            # Find corresponding layer in SuperPoint
            for sp_layer in self.super_point.layers:
                if type(sp_layer) == type(layer):
                    sp_layer.set_weights(layer.get_weights())
                    print(f"Transferred weights for layer: {type(layer).__name__}")