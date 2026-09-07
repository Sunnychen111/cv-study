#include <iostream>
#include <opencv2/opencv.hpp>

int main()
{
    cv::Mat image =
        cv::Mat::zeros(480, 640, CV_8UC3);

    cv::rectangle(
        image,
        cv::Point(100, 100),
        cv::Point(300, 300),
        cv::Scalar(0, 255, 0),
        2);

    cv::imwrite("opencv_test.jpg", image);

    std::cout
        << "Image shape: "
        << image.cols
        << " x "
        << image.rows
        << std::endl;

    return 0;
}