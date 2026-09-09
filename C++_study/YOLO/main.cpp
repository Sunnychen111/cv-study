#include <iostream>
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
using namespace std;

int main()
{
    cout << "ONNX Runtime start" << endl;

    // 1. 创建 ONNX Runtime 环境
    Ort::Env env(
        ORT_LOGGING_LEVEL_WARNING,
        "YOLO");

    // 2. Session配置
    Ort::SessionOptions session_options;
    session_options.SetInterOpNumThreads(1);

    session_options.SetGraphOptimizationLevel(
        GraphOptimizationLevel::ORT_ENABLE_ALL);

    // 3.加载模型
    Ort::Session session(
        env,
        L"yolo11n.onnx",
        session_options);
    cout << "Model loaded successfully" << endl;

    // 4.输入输出设置
    // 保持和onnx的一致，目前是静态的
    size_t input_count = session.GetInputCount();
    size_t output_count = session.GetOutputCount();

    cout << "Input Size: " << input_count << endl;
    cout << "Output Size " << output_count << endl;

    Ort::AllocatorWithDefaultOptions allocator;

    // 5.输入信息
    for (size_t i = 0; i < input_count; i++)
    {
        auto input_name = session.GetInputNameAllocated(
            i,
            allocator);
        cout << "Input" << i << "name" << input_name.get() << endl;
    }

    for (size_t i = 0; i < output_count; i++)
    {
        auto output_name =
            session.GetOutputNameAllocated(
                i,
                allocator);

        cout << "Output " << i
             << " name: "
             << output_name.get()
             << endl;

        auto type_info = session.GetInputTypeInfo(i);
        auto tensor_info = type_info.GetTensorTypeAndShapeInfo();

        vector<int64_t> shape = tensor_info.GetShape();

        cout << "shape: " << endl;
        for (const auto &dim : shape)
        {
            cout << dim << " ";
        }
        cout << endl;
    }
    return 0;
}