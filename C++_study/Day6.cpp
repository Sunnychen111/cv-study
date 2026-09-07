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

struct AssociationResult
{
    vector<pair<int, int>> matches;
    vector<int> unmatched_tracks;
    vector<int> unmatched_detection;

    AssociationResult() {};
};

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

// 这个threshold是iou的阈值
vector<vector<float>> Gate(vector<vector<float>> cost_matrix, float threshold)
{
    vector<vector<float>> applymatrix = cost_matrix;
    for (int i = 0; i < cost_matrix.size(); i++)
    {
        for (int j = 0; j < cost_matrix[0].size(); j++)
        {
            if (cost_matrix[i][j] >= 1 - threshold)
            {
                applymatrix[i][j] = 1e6;
            }
        }
    }
    return applymatrix;
}

pair<int, int> find_min(vector<bool> &detection, vector<bool> &track, const vector<vector<float>> &gate)
{
    float min = 1e6;
    int de = -1, tr = -1;
    pair<int, int> match = {-1, -1};
    for (int i = 0; i < gate.size(); i++)
    {
        for (int j = 0; j < gate[0].size(); j++)
        {
            if (gate[i][j] < min && detection[j] == false && track[i] == false)
            {
                min = gate[i][j];
                de = j;
                tr = i;
            }
        }
    }
    if (de != -1 && tr != -1)
    {
        detection[de] = true;
        track[tr] = true;
        match.first = tr;
        match.second = de;
    }
    return match;
}

// 全局贪心匹配
AssociationResult greedymatch(vector<vector<float>> gate)
{
    AssociationResult results;
    vector<pair<int, int>> greedymatch;
    pair<int, int> match;
    vector<bool> tracks(gate.size(), false);
    vector<bool> detection(gate[0].size(), false);
    for (int i = 0; i < gate.size(); i++)
    {
        match = find_min(detection, tracks, gate);
        if (match.first == -1 || match.second == -1)
        {
            break;
        }
        greedymatch.push_back(match);
    }
    results.matches = greedymatch;
    vector<int> unmatched_resultes;
    int i = 0;
    for (const auto &track : tracks)
    {
        if (track == false)
        {
            unmatched_resultes.push_back(i);
        }
        i++;
    }
    results.unmatched_tracks = unmatched_resultes;
    vector<int> unmatched_de;
    int j = 0;
    for (const auto &de : detection)
    {
        if (de == false)
        {
            unmatched_de.push_back(j);
        }
        j++;
    }
    results.unmatched_detection = unmatched_de;

    return results;
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

    float threshold = 0.3f; // 可以进行跟踪的iou阈值

    vector<vector<float>> apply = Gate(cost, threshold);
    cout << "ApplyGate" << endl;
    for (int i = 0; i < apply.size(); i++)
    {
        for (int j = 0; j < apply[0].size(); j++)
        {
            cout << apply[i][j] << " ";
        }
        cout << endl;
    }

    vector<vector<float>> gate =
        {
            {0.319328f, 1e6f, 1e6f},
            {1e6f, 0.529412f, 1e6f}};

    AssociationResult results = greedymatch(gate);

    for (const auto &match : results.matches)
    {
        cout << "Track "
             << match.first
             << " -> Detection "
             << match.second
             << endl;
    }

    cout << "Unmatches Tracks:" << endl;
    if (results.unmatched_tracks.size() == 0)
    {
        cout << "None" << endl;
    }
    else
    {
        for (const auto &track : results.unmatched_tracks)
        {
            cout << track << " ";
        }
        cout << endl;
    }

    cout << "Unmatches Detections:" << endl;
    if (results.unmatched_detection.size() == 0)
    {
        cout << "None" << endl;
    }
    else
    {
        for (const auto &detection : results.unmatched_detection)
        {
            cout << detection << " ";
        }
        cout << endl;
    }

    return 0;
}