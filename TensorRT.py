import onnxruntime as ort
import time
import tensorrt
import os

ort.preload_dlls()

# TensorRT 缓存机制
cache_path = os.path.abspath("./trt_cache")
os.makedirs(cache_path,exist_ok=True)

trt_options = {
    "device_id" : 0,

    # 保存构建好的TensorRT Engine （下次可以直接跳过build的阶段）
    "trt_option_cache_enable": True,
    "trt_option_cache_path" : cache_path,

    # 保存缓存构建的Timing信息 (记住哪些Kernel会更快)
    "trt_timing_cache_profile" : True,
    "trt_timing_cache_path" : cache_path,
}

start = time.perf_counter()
session = ort.InferenceSession(
    "Simple_model.onnx",
    providers=[
        "TensorrtExecutionProvider",
        "CUDAExecutionProvider"
    ]
)
end = time.perf_counter()

print("Session Provider")
print(session.get_providers())

print(
    "session prof_time:",
    (end-start)*1000,
    "ms"
)


"""
(cv-study) PS  python TensorRT.py
Session Provider
['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
session prof_time: 11111.59520000001 ms
(cv-study) PS python TensorRT.py
Session Provider
['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
session prof_time: 4939.493200000015 ms
(cv-study) PS  python TensorRT.py
Session Provider
['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
session prof_time: 4511.418200000207 ms
(cv-study) PS  python TensorRT.py
Session Provider
['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
session prof_time: 4478.306599999996 ms
"""