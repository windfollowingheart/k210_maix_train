# from train.detector.yolo.backend.decoder import YoloDecoder

# import tensorflow as tf
# import cv2
# import numpy as np
# import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--image_path", type=str, help="path to image file", default="")
parser.add_argument("--model_path", type=str, help="path to image file", default="")

args = parser.parse_args()
model_path=args.model_path
image_path=args.image_path
print(model_path)
print(image_path)
exit()

# 加载模型
# model_path=r'D:\committers-2022-06\pythonworkplace\AI\k210-master\out2\m_best.h5'
# image_path=r'D:\committers-2022-06\pythonworkplace\AI\k210-master\out\datasets\xml_format\images\0\58.jpg'
model = tf.keras.models.load_model(model_path)

# 加载图片
img = cv2.imread(image_path)
img1 = img.copy()
img = cv2.resize(img, (224, 224))
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = np.expand_dims(img, axis=0).astype(np.float32) / 255.0


# 进行推理
pred = model.predict(img)
netout1=pred[0]
print(netout1.shape)


anchors=[2.59375, 2.53125, 4.375, 4.28125, 1.734375, 1.890625, 3.4375, 3.46875, 5.25, 5.28125]
yolo_decoder = YoloDecoder(anchors)
boxes, probs = yolo_decoder.run(netout1, 0.3)

print(boxes)
print("==================")

print(probs)


x=boxes[0][0]*224
y=boxes[0][1]*224
w=boxes[0][2]*224
h=boxes[0][3]*224

x1=x-w/2
y1=y-h/2
x2=x1+w
y2=y1+h
print(x1)
print(x2)
print(y1)
print(y2)
cv2.rectangle(img1, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
confidence=int(np.max(probs[0])*100)/100
print(confidence)
text="person"

cv2.putText(img1, "{}:{:.2f}".format(text,confidence), (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
cv2.imshow('img',img1)
cv2.waitKey(0)
        