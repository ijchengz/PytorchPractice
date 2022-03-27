import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import cv2

# https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2
# 將 content_image 轉成 style_image 的樣式,先用 CV2 載入兩個圖樣
content_image = cv2.imread('data/images/03.jpg')
content_image = cv2.resize(content_image, (0, 0), None, 0.5, 0.5)
cv2.imshow("original", content_image)
content_image = cv2.cvtColor(content_image, cv2.COLOR_BGR2RGB)
style_image = cv2.imread('data/images/ss01.jpg')
cv2.imshow("style", cv2.resize(style_image, (0, 0), None, 0.5, 0.5))

# 使用 numpy 來轉成 float32 numpy array，加上批次維度，並正規化到[0,1]的區間大小
content_image = content_image.astype(np.float32)[np.newaxis, ...] / 255.
style_image = style_image.astype(np.float32)[np.newaxis, ...] / 255.

# 將 style image 轉成 256 pixels (模式訓練時用的大小), content image 可以任意大小
style_image = tf.image.resize(style_image, (256, 256))

# 載入 stylization 模式
# model = hub.load('https://tfhub.dev/google/magenta/arbitrary-imagestylization-v1-256/2')
model = hub.load('model')

# 得到結果並 show 出來
out_image = model(tf.constant(content_image), tf.constant(style_image))
img = cv2.cvtColor(np.squeeze(out_image), cv2.COLOR_RGB2BGR)
cv2.imshow("result", img)
cv2.waitKey(0)