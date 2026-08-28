import numpy as np
import torch 
import cv2

# image = np.random.rand(480,680,3)
# print(image.shape)
# image_sw=np.transpose(image,(2,0,1))
# print(image_sw.shape)
# tensor = torch.from_numpy(image_sw)
# x=tensor.unsqueeze(0)
# print(x.squeeze(0).shape)


# x = torch.arange(24)
# x = x.reshape(2,3,4)
# print(x)
# print(x.shape)

# x_reshape = x.reshape(3,2,4)
# print(x_reshape)
# print(x_reshape.shape)

# y = torch.arange(24).reshape(2,3,4)
# y_permute = y.permute(1,0,2)
# print(y_permute)
# print(y_permute.shape)


# image = np.random.randint(
#     0,
#     256,
#     (720, 1280, 3),
#     dtype=np.uint8
# )

# tensor = torch.from_numpy(image)
# x=tensor.permute(2,0,1).unsqueeze(0).float()/255
# print(x.size)
# print(x.dtype)
# print(x.min())
# print(x.max())

# Tensor 索引

# x = torch.randn(4, 3, 224, 224)

# a=x[0]
# b=x[0,0]
# c=x[:,0]
# d=x[:,:,0:100,0:100]
# print(d.shape)


# Boolean Mask
# scores = torch.tensor([
#     0.91,
#     0.23,
#     0.75,
#     0.41,
#     0.88
# ])
# mask = scores>=0.5
# filter_scores = scores[
#     mask==True
# ]
# print(mask)
# print(filter_scores)


# 过滤掉低置信度的框
# detections = torch.tensor([
#     [10, 20, 100, 120, 0.95, 0],
#     [30, 40,  80,  90, 0.32, 1],
#     [50, 60, 150, 180, 0.82, 2],
#     [20, 30,  70, 110, 0.21, 0]
# ])

# box = detections[:,0:4]
# confidence = detections[:,4]
# id=detections[:,5]
# mask = (confidence>=0.5) & (id!=2)
# filter_detections = detections[mask]
# print(box.shape)
# print(confidence.shape)
# print(mask)
# print(filter_detections)


#Day1 总结小题
detections = torch.tensor([
    [10, 20, 100, 120, 0.95, 0],
    [30, 40,  80,  90, 0.32, 1],
    [50, 60, 150, 180, 0.82, 2],
    [20, 30,  70, 110, 0.71, 0],
    [15, 25,  90, 130, 0.65, 1]
])

confidence = detections[:,4]
mask1=confidence>=0.6
res1=detections[mask1]
print(res1)

class_ids = detections[:,5]
mask2 = mask1 & (class_ids!=2)
res2=detections[mask2]
print(res2)

boxes =res2[:,:4]
scores = confidence
class_ids = class_ids.long()
print(boxes.shape)
print(scores.shape)
print(class_ids.dtype)


