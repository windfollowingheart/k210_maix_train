import tensorflow as tf
from tensorflow.keras.models import load_model
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--h5_path", type=str, help="path to image file", default="")
parser.add_argument("--tflite_path", type=str, help="path to image file", default="")

args = parser.parse_args()
h5_path=args.h5_path
tflite_path=args.tflite_path

# h5_path = "/kaggle/working/k210_maix_train/out/m_best.h5"
# tflite_path = "/kaggle/working/k210_maix_train/out/m.tflite"
# 加载模型
model = load_model(h5_path)
tf.compat.v1.disable_eager_execution()

converter = tf.compat.v1.lite.TFLiteConverter.from_keras_model_file(h5_path,
                                    output_arrays=['{}/BiasAdd'.format(model.get_layer(None, -2).name)])
tfmodel = converter.convert()
with open (tflite_path , "wb") as f:
    f.write(tfmodel)