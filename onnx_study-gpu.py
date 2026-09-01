import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn


# # ============================================================
# # 1. Define a simple PyTorch model
# # ============================================================

class SimpleModel(nn.Module):
    """
    一个最简单的 CNN 示例模型。

    Data flow:
        Input [B, 3, H, W]
            ↓
        Conv2d: 3 -> 8
            ↓
        ReLU
            ↓
        AdaptiveAvgPool2d(1, 1)
            ↓
        Flatten
            ↓
        Linear: 8 -> 3
            ↓
        Output [B, 3]
    """

    def __init__(self):
        super().__init__()

        # 输入通道为 3（RGB），输出通道为 8
        # kernel_size=3, padding=1, stride=1
        # 因此 H、W 不发生变化
        self.conv = nn.Conv2d(
            in_channels=3,
            out_channels=8,
            kernel_size=3,
            padding=1
        )

        # ReLU 激活函数：
        # ReLU(x) = max(0, x)
        # 用于给网络引入非线性
        self.relu = nn.ReLU()

        # 将每个 channel 的 H x W 特征压缩到 1 x 1
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # 输入 8 维特征，输出 3 维结果
        self.fc = nn.Linear(8, 3)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = self.pool(x)

        # [B, 8, 1, 1] -> [B, 8]
        x = x.flatten(1)

        x = self.fc(x)

        return x


# # ============================================================
# # 2. PyTorch inference
# # ============================================================

# # 固定随机种子，方便重复运行时观察结果
# torch.manual_seed(42)

model = SimpleModel()
model.eval()

# # Dummy / Example Input
# # 它不是训练数据，而是用于：
# # 1. 测试 PyTorch forward
# # 2. 帮助 ONNX exporter 确定模型输入和计算路径
# x = torch.randn(1, 3, 224, 224)

# with torch.inference_mode():
#     pytorch_output = model(x)

# print("===== PyTorch Inference =====")
# print("input shape :", x.shape)
# print("output shape:", pytorch_output.shape)
# print("output:")
# print(pytorch_output)


# # ============================================================
# # 3. Export PyTorch model to ONNX
# # ============================================================

# onnx_path = "simple_model.onnx"


# # 转换成onnx

# torch.onnx.export(
#     model,                      # PyTorch model
#     x,                          # example / dummy input
#     onnx_path,                  # output ONNX file
#     input_names=["input"],      # ONNX input tensor name
#     output_names=["output"],    # ONNX output tensor name
#     dynamo=True,   
#     dynamic_shapes={
#         "x":{
#             0:"batch_size",
#             2:"height",
#             3:"width"
#         }
#     },                 
#     opset_version=18,            # ONNX operator set version
#     external_data=False
# )


# print("\nONNX model exported to:", onnx_path)



# # ============================================================
# # 4. Check ONNX model
# # ============================================================

# onnx_model = onnx.load(onnx_path)

# # 检查 ONNX Graph 是否满足 ONNX 规范
# onnx.checker.check_model(onnx_model)

# print("ONNX model is OK")


# # ============================================================
# # 5. Inspect ONNX Graph
# # ============================================================

# graph = onnx_model.graph

# print("\n===== Graph Information =====")
# print("Graph name:", graph.name)


# # ------------------------------------------------------------
# # ONNX Inputs
# # ------------------------------------------------------------

# print("\n===== Inputs =====")

# for input_tensor in graph.input:
#     print(input_tensor.name)


# # ------------------------------------------------------------
# # ONNX Outputs
# # ------------------------------------------------------------

# print("\n===== Outputs =====")

# for output_tensor in graph.output:
#     print(output_tensor.name)


# # ------------------------------------------------------------
# # ONNX Nodes / Operators
# # ------------------------------------------------------------

# print("\n===== Nodes =====")

# for node in graph.node:
#     print(
#         "Operator:", node.op_type,
#         "| Input:", list(node.input),
#         "| Output:", list(node.output)
#     )

# """
# 本模型通常可以观察到：

# PyTorch                     ONNX
# --------------------------------------------------
# nn.Conv2d                  -> Conv
# nn.ReLU                    -> Relu
# AdaptiveAvgPool2d((1,1))   -> GlobalAveragePool
# flatten(1)                 -> Flatten
# nn.Linear                  -> Gemm

# 注意：
# PyTorch Layer 和 ONNX Operator 的名字不一定一一相同，
# ONNX 关注的是“计算语义”，而不是 Python API 名称。
# """


# # ------------------------------------------------------------
# # ONNX Initializers / Weights
# # ------------------------------------------------------------

# print("\n===== Initializers / Weights =====")

# for initializer in graph.initializer:
#     print(
#         initializer.name,
#         list(initializer.dims)
#     )

# """
# 预期类似：

# conv.weight [8, 3, 3, 3]
# conv.bias   [8]
# fc.weight   [3, 8]
# fc.bias     [3]

# conv.weight:
#     [out_channels, in_channels, kernel_h, kernel_w]

# fc.weight:
#     [out_features, in_features]

# Initializer 可以理解为模型已经训练/初始化好的固定参数。
# 它们会被保存到 ONNX 模型中。
# """


# # ------------------------------------------------------------
# # ONNX Opset
# # ------------------------------------------------------------

# print("\n===== Opset =====")

# for opset in onnx_model.opset_import:
#     print(
#         "Domain:", opset.domain,
#         "Version:", opset.version
#     )


# # ============================================================
# # 6. ONNX Runtime inference
# # ============================================================

# session = ort.InferenceSession(
#     onnx_path,
#     providers=["CPUExecutionProvider"]
# )

# input_info = session.get_inputs()[0]
# output_info = session.get_outputs()[0]

# print("\n===== ONNX Runtime Model Information =====")

# print("Input name :", input_info.name)
# print("Input shape:", input_info.shape)
# print("Input type :", input_info.type)

# print("Output name :", output_info.name)
# print("Output shape:", output_info.shape)
# print("Output type :", output_info.type)


# # ============================================================
# # 7. Convert PyTorch Tensor to NumPy
# # ============================================================

# # ONNX Runtime Python API 通常使用 NumPy ndarray 作为输入
# #
# # PyTorch:
# #     torch.Tensor
# #
# # ONNX Runtime:
# #     numpy.ndarray
# #
# # 如果 Tensor 在 GPU 上：
# #     x.detach().cpu().numpy()
# x_numpy = x.cpu().numpy()


# # ============================================================
# # 8. Run ONNX Runtime
# # ============================================================

# input_name = input_info.name
# output_name = output_info.name

# onnx_outputs = session.run(
#     [output_name],
#     {
#         input_name: x_numpy
#     }
# )

# # session.run() 返回 list
# # 即使模型只有一个 output，外层依然是 list
# onnx_output = onnx_outputs[0]

# print("\n===== ONNX Runtime Inference =====")

# print("session.run() type:")
# print(type(onnx_outputs))

# print("single output type:")
# print(type(onnx_output))

# print("ONNX output shape:")
# print(onnx_output.shape)

# print("ONNX output:")
# print(onnx_output)


# # ============================================================
# # 9. Compare PyTorch and ONNX Runtime outputs
# # ============================================================

# pytorch_output_numpy = (
#     pytorch_output
#     .detach()
#     .cpu()
#     .numpy()
# )

# print("\n===== Output Comparison =====")

# print("PyTorch output:")
# print(pytorch_output_numpy)

# print("\nONNX Runtime output:")
# print(onnx_output)


# # np.allclose 用于判断两个浮点数组是否在允许误差范围内足够接近
# is_close = np.allclose(
#     pytorch_output_numpy,
#     onnx_output,
#     rtol=1e-5,
#     atol=1e-6
# )

# print("\nPyTorch and ONNX outputs are close:", is_close)

"""session = ort.InferenceSession(
    "simple_model.onnx",
    provider_options=["CPUExecutionProvider"]
)

input_name = session.get_inputs()[0].name
print("Model Shape")
print(session.get_inputs()[0].shape)

x2 = np.random.randn(
    1,3,320,320
).astype(np.float32)

x3 = np.random.randn(
    8,3,224,224
).astype(np.float32)

print("Input_type")
print(x2.shape)

output=session.run(
    None,
    {
        input_name:x2
    }
)

output2 = session.run(
    None,
    {
        input_name:x3
    }
)

print(output)
print(output2)

print(ort.get_available_providers())
print(session.get_providers())"""

import time 

warms_up = 50 # 预热20次
test_run = 1000
x2 = np.random.randn(
    1,3,320,320
).astype(np.float32)

x2_gpu = ort.OrtValue.ortvalue_from_numpy(
    x2,
    "cuda",
    0
)   #将x2从CPU提前放到CUDA上


session = ort.InferenceSession(
    "simple_model.onnx",
    providers=["CUDAExecutionProvider"]
)


input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

io_blinding = session.io_binding()
io_blinding.bind_ortvalue_input(
    input_name,
    x2_gpu
)

output_gpu = ort.OrtValue.ortvalue_from_shape_and_type(
    [1, 3],
    np.float32,
    "cuda",
    0
)
io_blinding.bind_ortvalue_output(
    output_name,
    output_gpu
)

for _ in range(warms_up):
    session.run_with_iobinding(io_blinding)
    output = io_blinding.get_outputs()

# benchmark
lantencies = []

for i in range(test_run):
    start = time.perf_counter()
    session.run_with_iobinding(io_blinding)
    output = io_blinding.get_outputs()
    end = time.perf_counter()
    lantencies.append( (end-start) *1000)

print("Output device:", output[0].device_name())


avg_lantency = np.mean(lantencies)
min_lantency = np.min(lantencies)
median_lantency = np.median(lantencies)
max_lantency = np.max(lantencies)

fps = 1000 / avg_lantency
p95_latency = np.percentile(lantencies, 95)
p99_latency = np.percentile(lantencies, 99)

print("onnx")
print(avg_lantency,min_lantency,max_lantency,median_lantency,p95_latency,p99_latency)

print(fps)

lantencies_pytorch =[]

x = torch.from_numpy(x2)
model = model.cuda()
x = x.cuda()
with torch.inference_mode():
    for _ in range(warms_up):
        output = model(x)
with torch.inference_mode():
    for _ in range(test_run):
        torch.cuda.synchronize()
        start = time.perf_counter()
        output = model(x)
        torch.cuda.synchronize()
        end = time.perf_counter()
        lantencies_pytorch.append((end-start)*1000)

avg_lantency_pytorch = np.mean(lantencies_pytorch)
min_lantency_pytorch = np.min(lantencies_pytorch)
max_lantency_pytorch = np.max(lantencies_pytorch)
median_lantency_pytorch = np.median(lantencies_pytorch)
p95_latency_pytorch = np.percentile(lantencies_pytorch,95)
p99_latency_pytorch = np.percentile(lantencies_pytorch,99)

print("pytorch")
print(avg_lantency_pytorch,min_lantency_pytorch,max_lantency_pytorch,median_lantency_pytorch,p95_latency_pytorch,p99_latency_pytorch)
print(1000/avg_lantency_pytorch)





