import cv2
import os

video_path = "2.mp4"
output_path = "img"
interval_seconds = 1.0  # 每隔两秒保存一帧

os.makedirs(output_path, exist_ok=True)

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    raise RuntimeError(f"无法打开视频：{video_path}")

fps = cap.get(cv2.CAP_PROP_FPS)

if fps <= 0:
    cap.release()
    raise RuntimeError("无法获取视频帧率")

frame_interval = max(1, round(fps * interval_seconds))

frame_index = 0
saved_count = 1

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # 在第 0、2、4、6……秒保存画面
    if frame_index % frame_interval == 0:
        save_path = os.path.join(
            output_path,
            f"{saved_count+47}.jpg"
        )

        success = cv2.imwrite(save_path, frame)

        if success:
            print(f"已保存：{save_path}")
            saved_count += 1
        else:
            print(f"保存失败：{save_path}")

    frame_index += 1

cap.release()

print(f"抽帧完成，共保存 {saved_count - 1} 张图片")