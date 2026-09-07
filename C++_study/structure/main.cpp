

#include <iostream>
#include <vector>

#include "Detection.h"
#include "Association.h"

using namespace std;

int main()
{
    vector<Detection> tracks;

    tracks.push_back(
        Detection(
            100, 100,
            200, 200,
            1.0f, 0));

    tracks.push_back(
        Detection(
            300, 300,
            400, 400,
            1.0f, 0));

    vector<Detection> detections;

    detections.push_back(
        Detection(
            110, 110,
            210, 210,
            0.9f, 0));

    detections.push_back(
        Detection(
            320, 320,
            420, 420,
            0.8f, 0));

    detections.push_back(
        Detection(
            500, 500,
            600, 600,
            0.85f, 0));

    auto iou_matrix =
        buildIoUMatrix(
            tracks,
            detections);

    auto cost_matrix =
        buildCostMatrix(
            iou_matrix);

    auto gated_matrix =
        applyGate(
            cost_matrix,
            0.3f);

    AssociationResult result =
        greedyMatch(
            gated_matrix);

    cout << "Matches:" << endl;

    for (const auto &match : result.matches)
    {
        cout
            << "Track "
            << match.first
            << " -> Detection "
            << match.second
            << endl;
    }

    cout << "Unmatched Tracks:" << endl;

    if (result.unmatched_tracks.empty())
    {
        cout << "None" << endl;
    }
    else
    {
        for (const auto &id :
             result.unmatched_tracks)
        {
            cout << id << " ";
        }

        cout << endl;
    }

    cout << "Unmatched Detections:" << endl;

    if (result.unmatched_detections.empty())
    {
        cout << "None" << endl;
    }
    else
    {
        for (const auto &id :
             result.unmatched_detections)
        {
            cout << id << " ";
        }

        cout << endl;
    }

    return 0;
}