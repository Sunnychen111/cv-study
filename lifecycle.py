detections = [
    1, 1, 1,
    0, 0,
    1, 1,
    0, 0, 0, 0,
    1
]

min_hits = 3
max_age = 3

track = {
    "track_id": 0,
    "hits": 0,
    "age": 0,
    "time_since_update": 0,
    "state": "Tentative"
}


for frame_idx, detected in enumerate(detections):

    if track["state"] == "Remove":
        track["hits"]=0
        track["age"]=0
        track["time_since_update"] = 0


    if detected == 1:

        track["age"]+=1
        track["hits"]+=1
        track["time_since_update"] = 0

        if track["state"] == "Remove":
            track["track_id"]+=1

        if track["hits"] >= min_hits:
            track["state"] = "Confirm"
        else:
            track["state"] = "Tentative"

    else:

        track["age"]+=1
        track["time_since_update"]+=1
        if track["state"] == "Confirm":
            track["state"] = "Lost"
        if track["time_since_update"] >= 3:
            track["state"] = "Remove"


    print(
        "Frame ",
        frame_idx,
        " | Det ",
        detected,
        " | ID ",
        track["track_id"],
        " | Hits ",
        track["hits"],
        " | Age ",
        track["age"],
        " | Miss",
        track["time_since_update"],
        " | State ",
        track["state"]
    )