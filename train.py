from ultralytics import YOLO

model = YOLO("yolo11n.pt")

model.train(
    data="person.yaml",
    epochs=50,
    imgsz=640,
    batch=8,

    degrees=15,     # 随机旋转 -15° ~ +15°
    translate=0.1,  # 随机平移
    scale=0.2,      # 随机缩放
    fliplr=0.5      # 50%概率左右翻转
)