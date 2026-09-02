#include <iostream>
#include <vector>
using namespace std;

struct Detection
{
public:
    Detection(float x1, float x2, float y1, float y2, float conf, int id);
    Detection() {}
    float x1;
    float x2;
    float y1;
    float y2;
    float conf;
    int class_id;
};

Detection::Detection(float x1, float x2, float y1, float y2, float conf, int id)
{
    this->x1 = x1;
    this->x2 = x2;
    this->y1 = y1;
    this->y2 = y2;
    this->conf = conf;
    this->class_id = id;
}

int main()
{
    vector<Detection> Detections;
    Detection d1 = Detection(100, 300, 200, 400, 0.91, 0);
    Detections.push_back(d1);
    Detection d2 = Detection(120, 280, 220, 390, 0.72, 2);
    Detection d3 = Detection(400, 520, 100, 280, 0.86, 0);
    Detections.push_back(d2);
    Detections.push_back(d3);

    // const auto &det 只读遍历，常用于循环，不进行复制
    for (const auto &det : Detections)
    {
        if (det.conf >= 0.8 && det.class_id == 0)
        {
            cout << "bbox:" << "[" << det.x1 << "," << det.y1 << "," << det.x2 << "," << det.y2 << " conf " << det.conf << "]" << endl;
        }
        // det.conf = 0.7 如用const auto & 这条代码就会发生报错
    }
    return 0;
}