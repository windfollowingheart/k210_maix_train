import pynvml
import tensorflow as tf
pynvml.nvmlInit()
gpu_num = pynvml.nvmlDeviceGetCount()
# check nvidia driver
gpus = tf.config.experimental.list_physical_devices('GPU')
print(f"[INFO]: gpu_num : {gpu_num}, gpus: {len(gpus)}")
print(tf.__version__)