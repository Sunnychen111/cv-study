import torch
import onnxruntime as ort

print("CUDA:",torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU",torch.cuda.get_device_name(0))

print("ORT available providers:")
print(ort.get_available_providers())