import cv2

# https://github.com/Kautenja/a-neural-algorithm-of-artistic-style
# https://github.com/opencv/opencv/blob/3.4.0/samples/dnn/fast_neural_style.py
# 加載模型
net = cv2.dnn.readNetFromTorch('model/the_scream.t7')
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)

# 讀取圖片
image = cv2.imread('data/images/03.jpg')
image = cv2.resize(image,(0, 0), None, 0.5,0.5)
cv2.imshow('Original image', image)
(h, w) = image.shape[:2]

# blob: binary large object
blob = cv2.dnn.blobFromImage(image, 1.0, (w, h), (103.939, 116.779, 123.680), swapRB=False, crop=False)

# 進行計算
net.setInput(blob)
out = net.forward()
out = out.reshape(3, out.shape[2], out.shape[3])
out[0] += 103.939
out[1] += 116.779
out[2] += 123.68
out /= 255
out = out.transpose(1, 2, 0)

# 輸出圖片
cv2.imshow('Styled image', out)
cv2.waitKey(0)