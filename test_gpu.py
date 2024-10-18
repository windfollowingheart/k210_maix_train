import pynvml
import tensorflow as tf
from tensorflow.python.util import keras_deps

pynvml.nvmlInit()
gpu_num = pynvml.nvmlDeviceGetCount()
# check nvidia driver
gpus = tf.config.experimental.list_physical_devices('CPU')
print(f"[INFO]: gpu_num : {gpu_num}, gpus: {len(gpus)}")
print(tf.__version__)
print(tf.test.is_gpu_available())
print(keras_deps.get_clear_session_function())

# def save_model(model, h5_path, tflite_path=None):
#     print("save .h5 weights file to :", h5_path)
#     model.save(h5_path, overwrite=True, include_optimizer=False)
#     if tflite_path:
#         print("save tfilte to :", tflite_path)
#         import tensorflow as tf
# #         print("ok1")
#         converter = tf.lite.TFLiteConverter.from_keras_model(model)
#         print(converter)
#         tflite_model = converter.convert()
#         print(tflite_model)
#         print("ok2")
#         with open (tflite_path, "wb") as f:
#             f.write(tflite_model)
#         print("ok")

#         ## kpu V3 - nncase = 0.1.0rc5
#         # model.save("weights.h5", include_optimizer=False)

# #         tf.compat.v1.disable_eager_execution()
# #         tf.compat.v1.enable_eager_execution()
#         print(111111111111111111111111)
# #         print(model)
# #         model.summary()
# #         converter = tf.compat.v1.lite.TFLiteConverter.from_keras_model_file(h5_path,
# #                                             output_arrays=['{}/BiasAdd'.format(model.get_layer(None, -2).name)])
# #         converter = tf.compat.v1.lite.TFLiteConverter.from_keras_model_file(h5_path)
# #         print(1111111111111111111111112222222222)
# #         tfmodel = converter.convert()
# #         with open (tflite_path , "wb") as f:
# #             f.write(tfmodel)
#         del tf

# h5_path = "/kaggle/working/m1.h5"
# tflite_path = "/kaggle/working/m1.tflite"
# save_model(model, h5_path, tflite_path)

