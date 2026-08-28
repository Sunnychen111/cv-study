import av
import cv2
from ultralytics import YOLO


rtsp_url = "rtsp://admin:ngsk0809@192.168.5.57:554/Streaming/Channels/101"

model = YOLO(
    r"runs\detect\train5\weights\best.pt"
)

save_path = "output_yolo2.mp4"


# 打开 RTSP
container = av.open(
    rtsp_url,
    options={"rtsp_transport": "tcp"}
)

video_stream = container.streams.video[0]

writer = None
saving = False

print("开始实时检测")
print("按 S 开始保存视频")
print("按 Q 退出")


try:

    for frame in container.decode(video=0):

        # PyAV -> OpenCV
        img = frame.to_ndarray(format="bgr24")

        # YOLO
        result = model(
            img,
            conf=0.3,
            imgsz=640,
            verbose=False
        )[0]

        # 带检测框
        result_img = result.plot()


        # =========================
        # 如果正在保存
        # =========================
        if saving:
            writer.write(result_img)


        # =========================
        # 缩小显示
        # =========================
        h, w = result_img.shape[:2]

        show_width = 960
        show_height = int(h * show_width / w)

        show_img = cv2.resize(
            result_img,
            (show_width, show_height)
        )

        # 显示是否正在录像
        if saving:
            cv2.putText(
                show_img,
                "REC",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2
            )

        cv2.imshow(
            "YOLO Person Detection",
            show_img
        )


        key = cv2.waitKey(1) & 0xFF


        # =========================
        # S 开始保存
        # =========================
        if key == ord("s") and not saving:

            h, w = result_img.shape[:2]

            writer = cv2.VideoWriter(
                save_path,
                cv2.VideoWriter_fourcc(*"mp4v"),
                20.0,
                (w, h)
            )

            saving = True

            print("开始保存视频:", save_path)


        # =========================
        # Q 退出
        # =========================
        if key == ord("q"):
            break


finally:

    if writer is not None:
        writer.release()

    container.close()

    cv2.destroyAllWindows()

    if saving:
        print("视频已保存:", save_path)