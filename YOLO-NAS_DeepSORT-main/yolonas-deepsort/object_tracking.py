import numpy as np
import datetime
import cv2
import torch
from absl import app, flags, logging
from absl.flags import FLAGS
from deep_sort_realtime.deepsort_tracker import DeepSort
from super_gradients.training import models
from super_gradients.common.object_names import Models

# Định nghĩa các tham số dòng lệnh
flags.DEFINE_string('model', 'yolo_nas_l', 'yolo_nas_l or yolo_nas_m or yolo_nas_s')
flags.DEFINE_string('video', './data/video/test.mp4', 'path to input video or set to 0 for webcam')
flags.DEFINE_string('output', './output/output.mp4', 'path to output video')
flags.DEFINE_float('conf', 0.50, 'confidence threshhold')
flags.DEFINE_integer('class_id', None, 'class id 0 for person check coco.names for others')
flags.DEFINE_integer('blur_id', None, 'class id to blurring the object')

def main(_argv):
    # Khởi tạo các đối tượng đọc video và ghi video
    video_cap = cv2.VideoCapture(FLAGS.video)
    frame_width = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(video_cap.get(cv2.CAP_PROP_FPS))

    # Khởi tạo đối tượng ghi video
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    writer = cv2.VideoWriter(FLAGS.output, fourcc, fps, (frame_width, frame_height))

    # Khởi tạo đối tượng theo dõi bằng DeepSORT
    tracker = DeepSort(max_age=50)

    # Kiểm tra nếu GPU có sẵn, nếu không sử dụng CPU
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    # Tải mô hình YOLO
    model = models.get(FLAGS.model, pretrained_weights="coco").to(device)

    # Tải các nhãn lớp COCO mà mô hình YOLO đã được huấn luyện
    classes_path = "./configs/coco.names"
    with open(classes_path, "r") as f:
        class_names = f.read().strip().split("\n")

    #  Tạo danh sách các màu ngẫu nhiên để đại diện cho mỗi lớp
    np.random.seed(42)  # để có cùng màu
    colors = np.random.randint(0, 255, size=(len(class_names), 3))  # (80, 3)

    while True:
        # Bắt đầu thời gian để tính FPS
        start = datetime.datetime.now()
        
        # Đọc một khung hình từ video
        ret, frame = video_cap.read()

        # Nếu không có khung hình, nghĩa là đã đến cuối video
        if not ret:
            print("End of the video file...")
            break

        # Chạy mô hình YOLO trên khung hình hiện tại

        # Thực hiện phát hiện đối tượng bằng mô hình YOLO trên khung hiện tại
        detect = next(iter(model.predict(frame, iou=0.5, conf=FLAGS.conf)))

        # Trích xuất tọa độ bounding box, điểm số độ tin cậy, và nhãn lớp từ kết quả phát hiện
        bboxes_xyxy = torch.from_numpy(detect.prediction.bboxes_xyxy).tolist()
        confidence = torch.from_numpy(detect.prediction.confidence).tolist()
        labels = torch.from_numpy(detect.prediction.labels).tolist()
        # Kết hợp tọa độ hộp giới hạn và điểm tin cậy vào một danh sách duy nhất
        concate = [sublist + [element] for sublist, element in zip(bboxes_xyxy, confidence)]
        # Kết hợp danh sách được ghép nối với nhãn vào danh sách dự đoán cuối cùng
        final_prediction = [sublist + [element] for sublist, element in zip(concate, labels)]

        # Khởi tạo danh sách các bounding box và điểm số độ tin cậy
        results = []

        # Lặp qua các kết quả phát hiện
        for data in final_prediction:
            # Extract the confidence (i.e., probability) associated with the detection
            confidence = data[4]

            # Lọc các phát hiện yếu bằng cách đảm bảo độ tin cậy lớn hơn ngưỡng tối thiểu và với class_id
            if FLAGS.class_id == None:
                if float(confidence) < FLAGS.conf:
                    continue
            else:
                if ((int(data[5] != FLAGS.class_id)) or (float(confidence) < FLAGS.conf)):
                    continue

            # Nếu độ tin cậy lớn hơn ngưỡng tối thiểu, vẽ bounding box lên khung hình
            xmin, ymin, xmax, ymax = int(data[0]), int(data[1]), int(data[2]), int(data[3])
            class_id = int(data[5])
            
            # Add the bounding box (x, y, w, h), confidence, and class ID to the results list
            results.append([[xmin, ymin, xmax - xmin, ymax - ymin], confidence, class_id])

        # Cập nhật tracker với các phát hiện mới
        tracks = tracker.update_tracks(results, frame=frame)
        
        # Lặp qua các track
        for track in tracks:
            # Nếu theo dõi không được xác nhận, hãy bỏ qua
            if not track.is_confirmed():
                continue

            # Lấy ID theo dõi và hộp giới hạn
            track_id = track.track_id
            ltrb = track.to_ltrb()  # tọa độ bounding box: left, top, right, bottom
            class_id = track.get_det_class() # ID của lớp đối tượng
            x1, y1, x2, y2 = int(ltrb[0]), int(ltrb[1]), int(ltrb[2]), int(ltrb[3])
            
            # Lấy màu sắc cho lớp đối tượng
            color = colors[class_id]
            B, G, R = int(color[0]), int(color[1]), int(color[2])
            
            #  Tạo text cho track ID và tên lớp
            text = str(track_id) + " - " + str(class_names[class_id])
            
            # Áp dụng làm mờ Gaussian Blur nếu cần
            if FLAGS.blur_id is not None and class_id == FLAGS.blur_id:
                if x1 < x2 and y1 < y2 and x1 >= 0 and y1 >= 0 and x2 <= frame.shape[1] and y2 <= frame.shape[0]:
                    frame[y1:y2, x1:x2] = cv2.GaussianBlur(frame[y1:y2, x1:x2], (99,99), 5)

            # Vẽ bounding box và text lên khung hình
            cv2.rectangle(frame, (x1, y1), (x2, y2), (B, G, R), 2)
            cv2.rectangle(frame, (x1 - 1, y1 - 20), (x1 + len(text) * 12, y1), (B, G, R), -1)
            cv2.putText(frame, text, (x1 + 5, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # Kết thúc thời gian để tính FPS
        end = datetime.datetime.now()
        
        #  Hiển thị thời gian để xử lý 1 khung hình
        print(f"Time to process 1 frame: {(end - start).total_seconds() * 1000:.0f} milliseconds")
        
        # Tính toán FPS và vẽ nó lên khung hình
        fps = f"FPS: {1 / (end - start).total_seconds():.2f}"
        cv2.putText(frame, fps, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 8)

        cv2.imshow("Frame", frame)
        
        writer.write(frame)
        
        #  Kiểm tra xem phím 'q' có được nhấn để thoát khỏi vòng lặp không
        if cv2.waitKey(1) == ord("q"):
            break

    #Giải phóng các đối tượng video capture và video writer
    video_cap.release()
    writer.release()

    # Đóng tất cả các cửa sổ
    cv2.destroyAllWindows()

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
