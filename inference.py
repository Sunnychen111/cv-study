# import torch
# import torch.nn as nn

# device = torch.device(
#     "cuda" if torch.cuda.is_available() else "cpu"
# )


# class SimpleModel(nn.Module):
#     def __init__(self):
#         super().__init__()

#         self.conv = nn.Conv2d(
#             in_channels=3,
#             out_channels=8,
#             kernel_size=3,
#             padding=1
#         )

#         self.pool = nn.AdaptiveAvgPool2d((1, 1))

#         self.fc = nn.Linear(8, 3)

#     def forward(self, x):
#         x = self.conv(x)
#         print("after conv:", x.shape)
#         x = self.pool(x)
#         print("after pool:", x.shape)
#         x = x.flatten(1)
#         print("after squeeze:", x.shape)
#         x = self.fc(x)

#         return x

# model = SimpleModel()
# model = model.to(device)
# x = torch.randn(1, 3, 224, 224)
# x=x.to(device)
# model.eval()
# with torch.inference_mode():
#     output = model(x)

# print("input device:", x.device)
# print("model device:", next(model.parameters()).device)
# print("output device:", output.device)

# print(output.shape)


# Day2 综合练习
import torch
import torch.nn as nn
import numpy as np

device="cuda" if torch.cuda.is_available() else "cpu"


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv = nn.Conv2d(
            in_channels=3,
            out_channels=8,
            kernel_size=3,
            padding=1
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(8, 3)

    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)

        # [B,8,1,1] → [B,8]
        x = x.flatten(1) #展平

        x = self.fc(x)

        return x
    
# mission 1
image = np.random.randint(0,256,(480,640,3))


# mission 2
x = torch.from_numpy(image)
x=x.to(torch.uint8)

# mission 3
x=x.permute(2,0,1)


# mission 4
x= x.float()/255.0

print(x.min())
print(x.max())

# mission 5 
x=x.unsqueeze(0)


# mission 6
x = x.to(device)


# mission 9
model = SimpleModel()
model = model.to(device)
model.eval()
with torch.inference_mode():
    output = model(x)

print("input shape:", x.shape)
print("input dtype:", x.dtype)
print("input device:", x.device)

print("model device:", next(model.parameters()).device)

print("output shape:", output.shape)
print("output dtype:", output.dtype)
print("output device:", output.device)



