#include <iostream>
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include <string>
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

        auto output_info = session.GetOutputTypeInfo(i);
        auto output_tensor_info = output_info.GetTensorTypeAndShapeInfo();

        vector<int64_t> output_shape = output_tensor_info.GetShape();

        cout << "output_shape:" << endl;
        for (const auto &out : output_shape)
        {
            cout << out << " ";
        }
        cout << endl;
    }

    // 读取图片
    cv::Mat image;
    string image_path = "test3.jpg";
    image = cv::imread(image_path);

    if (image.empty())
    {
        cout << "image Fail to Read" << endl;
        return -1;
    }

    cv::Mat resized;
    cv::resize(
        image,
        resized,
        cv::Size(640, 640));

    cv::Mat rgb;
    cv::cvtColor(
        resized,
        rgb,
        cv::COLOR_BGR2RGB

    );

    cv::Mat float_img;
    rgb.convertTo(
        float_img,
        CV_32FC3,
        1.0 / 255.0

    );

    // img = img.transpose(2,0,1)
    // img = np.expand_dims(img,0)
    vector<float> input_tensor_values(
        1 * 3 * 640 * 640);
    vector<cv::Mat> channels(3);
    cv::split(float_img, channels);

    int channel_size = 640 * 640;

    memcpy(
        input_tensor_values.data(),
        channels[0].data,
        channel_size * sizeof(float));

    memcpy(
        input_tensor_values.data() + channel_size,
        channels[1].data,
        channel_size * sizeof(float));

    memcpy(
        input_tensor_values.data() + 2 * channel_size,
        channels[2].data,
        channel_size * sizeof(float));

    vector<int64_t> input_shape = {
        1, 3, 640, 640};

    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(
        OrtArenaAllocator,
        OrtMemTypeDefault);

    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info,
        input_tensor_values.data(),
        input_tensor_values.size(),
        input_shape.data(),
        input_shape.size());

    // 获取Input和Output 名字
    auto input_name = session.GetInputNameAllocated(
        0,
        allocator);

    auto output_name = session.GetOutputNameAllocated(
        0,
        allocator);

    const char *input_names[] = {input_name.get()};
    const char *output_names[] = {output_name.get()};

    // 正式进入推理
    // outputs = session.run(None,{"image":input_tensor})
    auto outputs = session.Run(
        Ort::RunOptions(nullptr),

        input_names,
        &input_tensor,
        1,

        output_names,
        1);

    float *output_data = outputs[0].GetTensorMutableData<float>();

    const int num_classes = 80; // YOLO默认权重80个分类
    const int num_candidates = 8400;

    cout << "Inference finished!" << endl;

    float global_max_score = 0.0f;
    int global_class = -1;
    int global_candidate = -1;

    for (int i = 0; i < num_candidates; i++)
    {
        for (int j = 0; j < num_classes; j++)
        {
            float score =
                output_data[(4 + j) * num_candidates + i];

            if (score > global_max_score)
            {
                global_max_score = score;
                global_class = j;
                global_candidate = i;
            }
        }
    }

    cout << "Global max score: "
         << global_max_score << endl;

    cout << "Best class: "
         << global_class << endl;

    cout << "Best candidate: "
         << global_candidate << endl;

    float conf_threshold = 0.5f;
    float nms_threshold = 0.45f;

    vector<cv::Rect> boxes;
    vector<float> scores;
    vector<int> classes_id;

    for (int i = 0; i < num_candidates; i++)
    {
        // bbox
        float cx = output_data[0 * num_candidates + i];
        float cy = output_data[1 * num_candidates + i];
        float w = output_data[2 * num_candidates + i];
        float h = output_data[3 * num_candidates + i];

        // 按80类中的最大置信度
        float max_score = 0.0f;
        int class_id = -1;

        for (int j = 0; j < num_classes; j++)
        {
            float score = output_data[(4 + j) * num_candidates + i];
            if (score > max_score)
            {
                max_score = score;
                class_id = j;
            }
        }

        // conf
        if (max_score < conf_threshold)
        {
            continue;
        }

        int x = static_cast<int>(cx - w / 2.0f);
        int y = static_cast<int>(cy - h / 2.0f);
        int width = static_cast<int>(w);
        int height = static_cast<int>(h);

        // 输出结果
        boxes.push_back(
            cv::Rect(x, y, width, height));

        scores.push_back(max_score);
        classes_id.push_back(class_id);
    }

    cout << "Candidates after confidence filter: "
         << boxes.size()
         << endl;

    // NMS 减少重复识别框
    vector<int> nms_indices;

    cv::dnn::NMSBoxes(
        boxes,
        scores,
        conf_threshold,
        nms_threshold,
        nms_indices);

    cout << "Detection num: " << nms_indices.size()
         << endl;

    for (const auto &idx : nms_indices)
    {
        const cv::Rect &box = boxes[idx];

        cout << "class_id: "
             << classes_id[idx]
             << " conf: "
             << scores[idx]
             << " bbox: ["
             << box.x << ", "
             << box.y << ", "
             << box.x + box.width << ", "
             << box.y + box.height << "]"
             << endl;
    }

    cv::resize(image, resized, cv::Size(640, 640));
    float scale_x = static_cast<float>(image.cols) / 640.0f;
    float scale_y = static_cast<float>(image.rows) / 640.0f;

    for (const auto &idx : nms_indices)
    {
        cv::Rect box = boxes[idx];

        int x1 =
            static_cast<int>(box.x * scale_x);

        int y1 =
            static_cast<int>(box.y * scale_y);

        int x2 =
            static_cast<int>(
                (box.x + box.width) * scale_x);

        int y2 =
            static_cast<int>(
                (box.y + box.height) * scale_y);

        cv::rectangle(
            image,
            cv::Point(x1, y1),
            cv::Point(x2, y2),
            cv::Scalar(0, 255, 0),
            2);

        string text =
            "id:" +
            to_string(classes_id[idx]) +
            " conf:" +
            to_string(scores[idx]);

        cv::putText(
            image,
            text,
            cv::Point(x1, max(0, y1 - 10)),
            cv::FONT_HERSHEY_SIMPLEX,
            0.6,
            cv::Scalar(0, 255, 0),
            2);

        cout << "class_id: "
             << classes_id[idx]
             << " conf: "
             << scores[idx]
             << " bbox: ["
             << x1 << ", "
             << y1 << ", "
             << x2 << ", "
             << y2 << "]"
             << endl;
    }

    cv::imwrite(
        "yolo_result.jpg",
        image);

    return 0;
}