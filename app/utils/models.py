import tensorflow as tf
from tensorflow.keras.models import load_model
import app.utils.config as config

# physical_devices = tf.config.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(physical_devices[0], True)

class MyMeanIOU(tf.keras.metrics.MeanIoU):
    def update_state(self, y_true, y_pred, sample_weight=None):
        return super().update_state(y_true, tf.argmax(y_pred, axis=-1), sample_weight)


sem_seg_model = load_model(config.cfg.phenot.path, custom_objects={'MyMeanIOU': MyMeanIOU})
