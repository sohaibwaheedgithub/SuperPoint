import keras
import tensorflow as tf
import numpy as np
from superpoint.visualization.utils.image_grid import images_to_grid


class TrainingImageLogger(keras.callbacks.Callback):
    def __init__(self, writer, images, points, epoch, max_outputs=4, pred_threshold=0.5):
        super().__init__()
        self._writer = writer
        self._images = tf.convert_to_tensor(images)
        self._points = tf.convert_to_tensor(points)
        self._epoch = epoch
        self._max_outputs = max_outputs
        self._pred_threshold = pred_threshold


    def _draw_points(self, image_rgb, points, color, radius=2):
        h, w, _ = image_rgb.shape
        for y, x in points.astype(np.int32):
            y0 = max(0, y - radius)
            y1 = min(h, y + radius + 1)
            x0 = max(0, x - radius)
            x1 = min(w, x + radius + 1)
            image_rgb[y0:y1, x0:x1] = color
        return image_rgb
   


    def on_epoch_end(self, epoch, logs=None):
        outputs = self.model(self._images, training=False)
        heatmaps = tf.clip_by_value(outputs["heatmap"], 0.0, 1.0)
        overlays = []
        max_count = min(self._max_outputs, int(self._images.shape[0]))

        for i in range(max_count):
            image = self._images[i].numpy()
            image = (np.clip(image, 0.0, 1.0) * 255).astype(np.uint8)
            if image.ndim == 2 or image.shape[-1] == 1:
                image = np.repeat(image[..., :1], 3, axis=-1)

            gt_points = self._points[i].numpy()
            gt_points = gt_points[np.any(gt_points != 0, axis=-1)]

            pred_mask = heatmaps[i, ..., 0].numpy() >= self._pred_threshold
            pred_y, pred_x = np.where(pred_mask)
            if pred_y.size:
                pred_points = np.stack([pred_y, pred_x], axis=-1)
            else:
                pred_points = np.empty((0, 2), dtype=np.int32)

            overlay = image.copy()
            overlay = self._draw_points(overlay, gt_points, color=np.array([0, 255, 0], dtype=np.uint8))
            overlay = self._draw_points(overlay, pred_points, color=np.array([255, 0, 0], dtype=np.uint8))
            overlays.append(overlay)

        overlays = tf.convert_to_tensor(np.stack(overlays, axis=0), dtype=tf.uint8)

            

        with self._writer.as_default():
            tf.summary.image(
                "visuals/heatmaps",
                heatmaps,
                step=self._epoch,
                max_outputs=self._max_outputs,
            )
            tf.summary.image(
                "overlays/gt_points_vs_pred_points",
                overlays,
                step=self._epoch,
                max_outputs=self._max_outputs,
            )

            # SEConvBlock_1_conv2d_1 Filter Logs

            for i in range(outputs["SEConvBlock_1_conv2d_1"].shape[0]):
                if i == self._max_outputs:
                    break
                
                acts = outputs["SEConvBlock_1_conv2d_1"][i:i+1]
                acts = tf.transpose(acts, [3, 1, 2, 0])   # [64, H, W, 1]

                # Normalize each activation map independently
                acts_min = tf.reduce_min(acts, axis=[1, 2], keepdims=True)
                acts_max = tf.reduce_max(acts, axis=[1, 2], keepdims=True)

                act_imgs = (acts - acts_min) / (acts_max - acts_min + 1e-8)

                # ---------------------------------------------------
                # Create a single 8x8 grid image with borders
                # ---------------------------------------------------
                
                grid = images_to_grid(act_imgs)

                # Log single image
                tf.summary.image(
                    f"SEConvBlock_1_conv2d_1/Activations/Sample {i+1}",
                    grid,
                    step=self._epoch,
                    max_outputs=1
                )

                # Log activation histogram
                tf.summary.histogram(
                    f"SEConvBlock_1_conv2d_1/Activations/Sample {i+1}",
                    acts,
                    step=self._epoch
                )

            # Log Kernel for SEConvBlock_1_conv2d_1
            filters = tf.transpose(self.model.encoder.SEConvBlock_1.conv2d_1.kernel, [3, 1, 2, 0])
            filters_grid = images_to_grid(filters)

            tf.summary.image(
                f"SEConvBlock_1_conv2d_1/Kernel",
                filters_grid,
                step=self._epoch
            )

            self._writer.flush()