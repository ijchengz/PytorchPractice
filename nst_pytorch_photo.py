import re
import cv2
import torch
from torchvision import transforms
from trans_net import TransformerNet

# https://github.com/pytorch/examples/tree/main/fast_neural_style
# 讀取圖片與模型
content_image = cv2.imread('data/images/03.jpg')
content_image = cv2.resize(content_image, (0, 0), None, 0.6,0.6)
content_image = cv2.cvtColor(content_image, cv2.COLOR_BGR2RGB)

style_model = TransformerNet()
state_dict = torch.load('model/udnie.pth')

for k in list(state_dict.keys()):
    if re.search(r'in\d+\.running_(mean|var)$', k):
        del state_dict[k]

style_model.load_state_dict(state_dict)
style_model.to('cpu')
style_model.eval()

# 進行計算
content_transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x.mul(255))])
content_image = content_transform(content_image)
content_image = content_image.unsqueeze(0).to('cpu')

with torch.no_grad():
    output = style_model(content_image).cpu()

# 輸出圖片
img = output[0].clamp(0, 255).numpy().transpose(1, 2, 0).astype("uint8")
img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

cv2.imshow("Styled images", img)
cv2.waitKey(0)