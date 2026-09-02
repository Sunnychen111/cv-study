import onnxruntime as ort
import time
import tensorrt
import os
import numpy as np

ort.preload_dlls()

# TensorRT 缓存机制
cache_path = os.path.abspath("./trt_cache")
os.makedirs(cache_path,exist_ok=True)

# provider options 没有显式显示FP16，就是使用FP32
trt_options = {
    "device_id" : 0,
    "trt_fp16_enable": False,

    # 保存构建好的TensorRT Engine （下次可以直接跳过build的阶段）
    "trt_engine_cache_enable": True,
    "trt_engine_cache_path" : cache_path,

    # 保存缓存构建的Timing信息 (记住哪些Kernel会更快)
    "trt_timing_cache_enable" : True,
    "trt_timing_cache_path" : cache_path,
}


start = time.perf_counter()
session = ort.InferenceSession(
    "Simple_model.onnx",
    providers=[
        ("TensorrtExecutionProvider",trt_options),
        "CUDAExecutionProvider"
    ]
)
end = time.perf_counter()


# print("Session Provider")
# print(session.get_providers())

# print(
#     "session prof_time:",
#     (end-start)*1000,
#     "ms"
# )


x2 = np.random.randn(1,3,320,320).astype(np.float32)
x2_gpu = ort.OrtValue.ortvalue_from_numpy(
    x2,
    "cuda",
    0
)
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name
io_blinding = session.io_binding()

output_gpu = ort.OrtValue.ortvalue_from_shape_and_type(
    [1, 3],
    np.float32,
    "cuda",
    0
)

io_blinding.bind_ortvalue_input(
    input_name,
    x2_gpu
)

io_blinding.bind_ortvalue_output(
    output_name,
    output_gpu
)



warms_up = 50
test_times = 1000
inference = []

for _ in range(warms_up):
    session.run_with_iobinding(io_blinding)
    

for _ in range(test_times):
    start = time.perf_counter()
    # output = session.run(
    #     None,
    #     {
    #         input_name : x2
    #     }
    # )
    
    session.run_with_iobinding(io_blinding)
    end = time.perf_counter()
    # output = io_blinding.get_outputs()
    inference.append ( (end-start)*1000)

print("avg_time:", np.average(inference))
print("min_time:",np.min(inference)),
print("max_time",np.max(inference)),
print("p95_time:",np.percentile(inference,95))
print("p99_time:",np.percentile(inference,99))
print("avg_fps:" ,1000/np.average(inference))





print(io_blinding.get_outputs()[0].numpy)
print(session.get_providers())


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