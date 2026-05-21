import cv2
import yt_dlp
import time
import csv
from datetime import datetime
from collections import deque, defaultdict
from ultralytics import YOLO

# Load YOLO model
model = YOLO('yolo11l.pt')
class_list = model.names

# Target FPS
target_fps = 10
frame_time = 1 / target_fps

# YouTube livestream
url = "https://youtu.be/z545k7Tcb5o"

ydl_opts = {
    "format": "best[height<=480]",
}

# Counting variables
class_counts = defaultdict(int)
crossed_ids = set()

# IoT tracking variables
start_time = time.time()
last_log_time = time.time()

# Emission weights (environmental impact)
EMISSION_WEIGHTS = {
    "car": 1,
    "motorcycle": 0.5,
    "bus": 4,
    "truck": 3
}

#def is_right_of_line(px, py, x1, y1, x2, y2):
 #   return (px - x1) * (y2 - y1) - (py - y1) * (x2 - x1) < 0

# CSV logging setup
csv_file = "traffic_data.csv"
with open(csv_file, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "timestamp",
        "cars",
        "trucks",
        "buses",
        "motorcycles",
        "vehicles_per_min",
        "traffic_level",
        "emission_index"
    ])

def log_data(data):
    with open(csv_file, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(data)

# Buffer
buffer = deque(maxlen=10)

# Get stream URL
with yt_dlp.YoutubeDL(ydl_opts) as ydl:
    info = ydl.extract_info(url, download=False)
    stream_url = info['url']

cap = cv2.VideoCapture(stream_url)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)

if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit()

while True:
    start = time.time()

    vehicles_in_frame = 0

    # Skip frames for speed
    for _ in range(3):
        cap.grab()

    ret, frame = cap.read()
    if not ret:
        break

    buffer.append(frame)

    if len(buffer) == 0:
        continue

    display_frame = buffer.pop()
    buffer.clear()

    # YOLO tracking
    results = model.track(
        display_frame,
        persist=True,
        tracker="bytetrack.yaml",
        classes=[2, 3, 5, 7]
    )

    boxes = results[0].boxes
    h, w = display_frame.shape[:2]
    roi_y = h // 4

    if boxes is not None and boxes.id is not None:
        xyxy = boxes.xyxy.cpu()
        track_ids = boxes.id.int().cpu().tolist()
        class_indices = boxes.cls.int().cpu().tolist()

        for box, track_id, class_idx in zip(xyxy, track_ids, class_indices):
            x1, y1, x2, y2 = map(int, box)
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            class_name = class_list[class_idx]

            # Draw
            cv2.circle(display_frame, (cx, cy), 4, (0, 0, 255), -1)
            cv2.putText(display_frame, f"ID:{track_id} {class_name}",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 255, 0), 2)
            cv2.rectangle(display_frame, (x1, y1), (x2, y2),
                          (0, 255, 0), 2)

            # COUNTING with ROI restriction
            if cy > roi_y:
                vehicles_in_frame += 1
                if track_id not in crossed_ids:
                    crossed_ids.add(track_id)
                    class_counts[class_name] += 1

    # IOT
    current_time = time.time()
    elapsed_minutes = (current_time - start_time) / 60

    total_vehicles = sum(class_counts.values())
    if elapsed_minutes > 0:
        vehicles_per_min = total_vehicles / elapsed_minutes
    else:
        vehicles_per_min = 0

    # Traffic level classification
    if vehicles_per_min < 30:
        traffic_level = "LOW"
    elif vehicles_per_min < 60:
        traffic_level = "MEDIUM"
    else:
        traffic_level = "HIGH"

    # emission index calculation
    emission_index = 0
    for vehicle, count in class_counts.items():
        emission_index += EMISSION_WEIGHTS.get(vehicle, 1) * count

    # event generation
    if vehicles_in_frame > 15:
        print("EVENT: Traffic congestion detected")
        cv2.putText(display_frame, "Traffic Congestion Detected!",
                    (w-300, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 255, 0), 2)

    if emission_index > 1000:
        print("EVENT: High pollution level detected")
        cv2.putText(display_frame, "High Pollution Level Detected!",
                    (w-300, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 255, 0), 2)



    # data logging every 10 secs
    if current_time - last_log_time > 10:
        log_data([
            datetime.now(),
            class_counts.get("car", 0),
            class_counts.get("truck", 0),
            class_counts.get("bus", 0),
            class_counts.get("motorcycle", 0),
            round(vehicles_per_min, 2),
            traffic_level,
            emission_index
        ])
        last_log_time = current_time


    # display
    cv2.line(display_frame, (0, roi_y), (w, roi_y), (255, 255, 0), 2)
    cv2.putText(display_frame, "Counting Zone",
                (10, roi_y + 14),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (255, 255, 0), 2)

    cv2.putText(display_frame, f"Traffic: {traffic_level}",
                (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (0, 255, 255), 2)

    cv2.putText(display_frame, f"Veh/min: {int(vehicles_per_min)}",
                (50, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (255, 255, 0), 2)

    cv2.putText(display_frame, f"Emission idx: {int(emission_index)}",
                (50, 110),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (0, 100, 255), 2)

    y_offset = 150
    for class_name, count in class_counts.items():
        cv2.putText(display_frame, f"{class_name}: {count}",
                    (50, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 0), 2)
        y_offset += 25

    cv2.imshow("Smart Traffic Monitoring", display_frame)

    elapsed = time.time() - start
    time.sleep(max(0, frame_time - elapsed))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()