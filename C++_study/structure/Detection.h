#pragma once

struct Detection
{
    float x1;
    float y1;
    float x2;
    float y2;
    float conf;
    int class_id;

    Detection() = default;

    Detection(
        float x1,
        float y1,
        float x2,
        float y2,
        float conf,
        int id)
        : x1(x1),
          y1(y1),
          x2(x2),
          y2(y2),
          conf(conf),
          class_id(id)
    {
    }
};