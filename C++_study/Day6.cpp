#include <iostream>
#include <vector>
#include <algorithm>
using namespace std;

struct Detection
{
public:
    Detection(float x1, float y1, float x2, float y2, float conf, int id);
    Detection() {};
    float x1;
    float x2;
    float y1;
    float y2;
    float conf;
    int class_id;
};

Detection::Detection(float x1, float y1, float x2, float y2, float conf, int id)
{
    this->x1 = x1;
    this->x2 = x2;
    this->y1 = y1;
    this->y2 = y2;
    this->conf = conf;
    this->class_id = id;
}

float getArea(const Detection &det) // C++ 定义函数调用det
{
    float Area = (det.x2 - det.x1) * (det.y2 - det.y1);
    return Area;
}

float Iou(const Detection &a, const Detection &b)
{
    float inner = 0, outer = 0;
    float inner_w = min(a.x2, b.x2) - max(a.x1, b.x1);
    float inner_h = min(a.y2, b.y2) - max(a.y1, b.y1);
    if (inner_h <= 0 || inner_w <= 0)
    {
        inner = 0;
    }
    else
        inner = inner_h * inner_w;

    outer = getArea(a) + getArea(b) - inner;

    float iou = inner / outer;
    return iou;
}

vector<vector<float>> iou_martix(vector<Detection> tracks, vector<Detection> detections)
{
    vector<vector<float>> iou_martix;
    for (int i = 0; i < tracks.size(); i++)
    {
        vector<float> row;
        for (int j = 0; j < detections.size(); j++)
        {
            row.push_back(Iou(tracks[i], detections[j]));
        }
        iou_martix.push_back(row);
    }
    return iou_martix;
}

vector<vector<float>> cost_matrix(vector<vector<float>> iou)
{
    vector<vector<float>> cost_martix;
    for (const auto &iou_row : iou)
    {
        vector<float> cost_row;

        for (const auto &value : iou_row)
        {
            cost_row.push_back(1.0f - value);
        }
        cost_martix.push_back(cost_row);
    }
    return cost_martix;
}

int main()
{
    // vector<Detection> Detections;
    // Detection d1 = Detection(100, 200, 300, 400, 0.91, 0);
    // Detections.push_back(d1);
    // Detection d2 = Detection(120, 220, 280, 390, 0.72, 2);
    // Detection d3 = Detection(400, 100, 520, 280, 0.86, 0);
    // Detections.push_back(d2);
    // Detections.push_back(d3);

    // // const auto &det 只读遍历，常用于循环，不进行复制
    // for (const auto &det : Detections)
    // {
    //     if (det.conf >= 0.8 && det.class_id == 0)
    //     {
    //         cout << "bbox:" << "[" << det.x1 << "," << det.y1 << "," << det.x2 << "," << det.y2 << " conf " << det.conf << "] ";
    //         cout << "Area:" << getArea(det) << endl;
    //     }
    //     // det.conf = 0.7 如用const auto & 这条代码就会发生报错
    // }

    Detection a(100, 100, 200, 200, 0.9, 0);

    // 1. 完全重合
    Detection b1(100, 100, 200, 200, 0.8, 0);

    // 2. 部分重合
    Detection b2(150, 150, 250, 250, 0.8, 0);

    // 3. 完全不重合
    Detection b3(300, 300, 400, 400, 0.8, 0);

    cout << Iou(a, b1) << endl;
    cout << Iou(a, b2) << endl;
    cout << Iou(a, b3) << endl;

    // 测试iou_matrix
    Detection t1(100, 100, 200, 200, 1.0, 0);
    Detection t2(300, 300, 400, 400, 1.0, 0);
    vector<Detection> tracks;
    tracks.push_back(t1);
    tracks.push_back(t2);

    Detection d1(110, 110, 210, 210, 0.9, 0);
    Detection d2(320, 320, 420, 420, 0.8, 0);
    Detection d3(500, 500, 600, 600, 0.85, 0);
    vector<Detection> detections;
    detections.push_back(d1);
    detections.push_back(d2);
    detections.push_back(d3);

    vector<vector<float>> matrix = iou_martix(tracks, detections);
    vector<vector<float>> cost = cost_matrix(matrix);
    for (int i = 0; i < tracks.size(); i++)
    {
        for (int j = 0; j < detections.size(); j++)
        {
            cout << matrix[i][j] << " ";
        }
        cout << endl;
    }

    cout << "Cost:" << endl;
    for (int i = 0; i < tracks.size(); i++)
    {
        for (int j = 0; j < detections.size(); j++)
        {
            cout << cost[i][j] << " ";
        }
        cout << endl;
    }
    return 0;
}