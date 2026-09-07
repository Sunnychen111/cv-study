
#include "Association.h"

#include <algorithm>

float getArea(const Detection &det)
{
    float w = std::max(0.0f, det.x2 - det.x1);
    float h = std::max(0.0f, det.y2 - det.y1);

    return w * h;
}

float IoU(
    const Detection &a,
    const Detection &b)
{
    float inner_w =
        std::min(a.x2, b.x2) -
        std::max(a.x1, b.x1);

    float inner_h =
        std::min(a.y2, b.y2) -
        std::max(a.y1, b.y1);

    inner_w = std::max(0.0f, inner_w);
    inner_h = std::max(0.0f, inner_h);

    float inner = inner_w * inner_h;

    float outer =
        getArea(a) +
        getArea(b) -
        inner;

    if (outer <= 0.0f)
    {
        return 0.0f;
    }

    return inner / outer;
}

std::vector<std::vector<float>> buildIoUMatrix(
    const std::vector<Detection> &tracks,
    const std::vector<Detection> &detections)
{
    std::vector<std::vector<float>> iou_matrix;

    for (const auto &track : tracks)
    {
        std::vector<float> row;

        for (const auto &det : detections)
        {
            row.push_back(
                IoU(track, det));
        }

        iou_matrix.push_back(row);
    }

    return iou_matrix;
}

std::vector<std::vector<float>> buildCostMatrix(
    const std::vector<std::vector<float>> &iou_matrix)
{
    std::vector<std::vector<float>> cost_matrix;

    for (const auto &iou_row : iou_matrix)
    {
        std::vector<float> cost_row;

        for (const auto &iou : iou_row)
        {
            cost_row.push_back(
                1.0f - iou);
        }

        cost_matrix.push_back(cost_row);
    }

    return cost_matrix;
}

std::vector<std::vector<float>> applyGate(
    const std::vector<std::vector<float>> &cost_matrix,
    float iou_threshold)
{
    std::vector<std::vector<float>> gated_matrix =
        cost_matrix;

    float cost_threshold =
        1.0f - iou_threshold;

    for (int i = 0; i < gated_matrix.size(); i++)
    {
        for (int j = 0; j < gated_matrix[i].size(); j++)
        {
            if (gated_matrix[i][j] > cost_threshold)
            {
                gated_matrix[i][j] = 1e6f;
            }
        }
    }

    return gated_matrix;
}

// 这个为greedy中的辅助函数，因此没有在.h中进行一个定义
static std::pair<int, int> findMin(
    std::vector<bool> &detection_used,
    std::vector<bool> &track_used,
    const std::vector<std::vector<float>> &gate)
{
    float min_cost = 1e6f;

    int best_track = -1;
    int best_detection = -1;

    for (int i = 0; i < gate.size(); i++)
    {
        for (int j = 0; j < gate[i].size(); j++)
        {
            if (track_used[i] ||
                detection_used[j])
            {
                continue;
            }

            if (gate[i][j] < min_cost)
            {
                min_cost = gate[i][j];

                best_track = i;
                best_detection = j;
            }
        }
    }

    if (best_track == -1 ||
        best_detection == -1)
    {
        return {-1, -1};
    }

    track_used[best_track] = true;
    detection_used[best_detection] = true;

    return {
        best_track,
        best_detection};
}

AssociationResult greedyMatch(
    const std::vector<std::vector<float>> &gate)
{
    AssociationResult result;

    if (gate.empty() || gate[0].empty())
    {
        return result;
    }

    std::vector<bool> track_used(
        gate.size(),
        false);

    std::vector<bool> detection_used(
        gate[0].size(),
        false);

    while (true)
    {
        auto match = findMin(
            detection_used,
            track_used,
            gate);

        if (match.first == -1)
        {
            break;
        }

        result.matches.push_back(match);
    }

    for (int i = 0; i < track_used.size(); i++)
    {
        if (!track_used[i])
        {
            result.unmatched_tracks.push_back(i);
        }
    }

    for (int i = 0; i < detection_used.size(); i++)
    {
        if (!detection_used[i])
        {
            result.unmatched_detections.push_back(i);
        }
    }

    return result;
}