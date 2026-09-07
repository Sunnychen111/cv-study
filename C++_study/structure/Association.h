#pragma once
#include <vector>
#include <utility>
#include "Detection.h"

struct AssociationResult
{
    std::vector<std::pair<int, int>> matches;
    std::vector<int> unmatched_tracks;
    std::vector<int> unmatched_detections;
};

float getArea(const Detection &det);

float IoU(
    const Detection &a,
    const Detection &b);

std::vector<std::vector<float>> buildIoUMatrix(
    const std::vector<Detection> &tracks,
    const std::vector<Detection> &detections);

std::vector<std::vector<float>> buildCostMatrix(
    const std::vector<std::vector<float>> &iou_matrix);

std::vector<std::vector<float>> applyGate(
    const std::vector<std::vector<float>> &cost_matrix,
    float iou_threshold);

AssociationResult greedyMatch(
    const std::vector<std::vector<float>> &gated_matrix);