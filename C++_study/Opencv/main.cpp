#include <iostream>
#include <opencv2/opencv.hpp>
using namespace std;

int main()
{
    // 读取图片
    cv::Mat image = cv::imread("test.png");

    if (image.empty())
    {
        cout << "Fail to load image" << endl;
        return -1;
    }

    // image.shape
    cout << "width:" << image.cols << endl;
    cout << "heigh:" << image.rows << endl;
    cout << "channels:" << image.channels() << endl;

    // Resize
    cv::Mat resized;
    cv::resize(
        image,
        resized,
        cv::Size(640, 640));

    // cv.BGR2RGB
    cv::Mat rgb;
    cv::cvtColor(
        resized,
        rgb,
        cv::COLOR_BGR2RGB);

    cv::imwrite("output.jpg", resized);
    cout << "Done!" << endl;

    // 读取视频
    cv::VideoCapture cap("test.mp4");
    if (!cap.isOpened())
    {
        cout << "Fail to cap the video!" << endl;
        return -1;
    }

    cout << "FPS:" << cap.get(cv::CAP_PROP_FPS) << endl;
    cout << "width:" << cap.get(cv::CAP_PROP_FRAME_WIDTH) << endl;
    cout << "heigh:" << cap.get(cv::CAP_PROP_FRAME_HEIGHT) << endl;

    cv::Mat frame;
    int frame_id = 0;
    while (true)
    {
        if (!cap.read(frame))
        {
            break;
        }
        frame_id++;
    }

    cout << "success read frame:" << frame_id << endl;
}