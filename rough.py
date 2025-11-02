import cv2
import easyocr
from ultralytics import YOLO
import re
import csv
from datetime import datetime
import os
import time ## NEW ##
import psutil ## NEW ##
from collections import deque ## NEW ##

## NEW: GPU Monitoring (optional, for NVIDIA GPUs) ##
try:
    import pynvml
    pynvml.nvmlInit()
    GPU_HANDLE = pynvml.nvmlDeviceGetHandleByIndex(0)
    NVIDIA_GPU_AVAILABLE = True
    print("NVIDIA GPU monitoring enabled.")
except Exception as e:
    NVIDIA_GPU_AVAILABLE = False
    print(f"NVIDIA GPU monitoring not available: {e}. GPU metrics will be skipped.")

# --- SETTINGS ---
MODEL_PATH = 'best.pt'
VIDEO_PATH = 'ved1.mp4'
DISPLAY_WIDTH = 960
CONFIDENCE_THRESHOLD = 0.5
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.7
FONT_THICKNESS = 2
BOX_COLOR = (0, 255, 0)
TEXT_COLOR = (255, 255, 255)
TEXT_BG_COLOR = (0, 0, 0)

# --- STABILITY AND ACCURACY SETTINGS ---
STABLE_DETECTION_THRESHOLD = 5

## NEW PERFORMANCE SETTING ##
PROCESS_EVERY_N_FRAMES = 3

# --- CSV file settings ---
CSV_FILE_PATH = 'detections_log.csv'
CSV_HEADER = ['Timestamp', 'License Plate']


# --- HELPER FUNCTIONS (UNCHANGED) ---
def preprocess_for_ocr(crop: cv2.typing.MatLike) -> cv2.typing.MatLike:
    """Pre-processes a license plate crop to improve OCR accuracy."""
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    scale_factor = 2
    new_w, new_h = int(gray.shape[1] * scale_factor), int(gray.shape[0] * scale_factor)
    resized = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(resized)
    return enhanced

def format_license_plate(text: str) -> str | None:
    """Cleans, validates, and formats the OCR text. Returns None if invalid."""
    plate = "".join(filter(str.isalnum, text.upper())).replace("IND", "")
    if not (8 <= len(plate) <= 10): return None
    if len(plate) >= 4:
        rto_code = plate[2:4]
        corrected_rto = rto_code.replace('I', '1').replace('Z', '2').replace('O', '0').replace('S', '5')
        plate = plate[:2] + corrected_rto + plate[4:]
    pattern = r'^([A-Z]{2})([0-9]{2})([A-Z]{1,3})([0-9]{4})$'
    match = re.match(pattern, plate)
    if match: return f"{match.group(1)} {match.group(2)} {match.group(3)} {match.group(4)}"
    return None

## NEW: Function to draw metrics on the frame ##
def draw_performance_metrics(frame, metrics):
    y_offset = 30
    for name, value in metrics.items():
        text = f"{name}: {value}"
        cv2.putText(frame, text, (10, y_offset), FONT, FONT_SCALE, TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)
        y_offset += 30
    return frame

# --- INITIALIZATION ---
try:
    model = YOLO(MODEL_PATH)
except Exception as e:
    print(f"Error loading YOLO model: {e}")
    exit()

print("Loading EasyOCR reader...")
reader = easyocr.Reader(['en'], gpu=True)
print("EasyOCR reader loaded.")
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print(f"Error: Could not open video file at {VIDEO_PATH}")
    exit()

plate_tracking_history = {}
saved_plates_set = set()
if not os.path.exists(CSV_FILE_PATH):
    with open(CSV_FILE_PATH, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(CSV_HEADER)
    print(f"Created new log file: {CSV_FILE_PATH}")


## NEW: Performance tracking variables ##
frame_counter = 0
last_detections = []
process_times = deque(maxlen=30) # Store times for the last 30 processed frames
current_process = psutil.Process()
# Initialize CPU usage calculation
current_process.cpu_percent(interval=None)
performance_metrics = {}

# --- PROCESSING LOOP ---
while True:
    ret, frame = cap.read()
    if not ret:
        print("End of video or failed to read frame.")
        break
    
    frame_counter += 1
    frame = cv2.rotate(frame, cv2.ROTATE_180)

    (original_height, original_width) = frame.shape[:2]
    aspect_ratio = original_width / original_height
    new_height = int(DISPLAY_WIDTH / aspect_ratio)
    frame_resized = cv2.resize(frame, (DISPLAY_WIDTH, new_height))

    # --- CORE LOGIC WITH FRAME SKIPPING ---
    if frame_counter % PROCESS_EVERY_N_FRAMES == 0:
        # --- HEAVY PROCESSING ON KEYFRAMES ---
        start_time = time.perf_counter() ## MODIFIED ##
        
        last_detections = []
        results = model.track(frame_resized, persist=True, tracker="bytetrack.yaml")

        if results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
            track_ids = results[0].boxes.id.cpu().numpy().astype(int)
            scores = results[0].boxes.conf.cpu().numpy()

            for box, track_id, score in zip(boxes, track_ids, scores):
                if score < CONFIDENCE_THRESHOLD:
                    continue

                x1, y1, x2, y2 = box
                plate_crop = frame_resized[y1:y2, x1:x2]
                if plate_crop.size == 0: continue

                try:
                    processed_crop = preprocess_for_ocr(plate_crop)
                    ocr_results = reader.readtext(processed_crop)
                    if not ocr_results: continue

                    plate_text = "".join([res[1] for res in ocr_results]).strip().upper()
                    final_text = format_license_plate(plate_text)
                    if not final_text: continue

                    if track_id not in plate_tracking_history: plate_tracking_history[track_id] = {}
                    current_counts = plate_tracking_history[track_id]
                    current_counts[final_text] = current_counts.get(final_text, 0) + 1

                    text_to_display = ""
                    if current_counts[final_text] >= STABLE_DETECTION_THRESHOLD:
                        if final_text not in saved_plates_set:
                            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            data_row = [timestamp, final_text]
                            with open(CSV_FILE_PATH, mode='a', newline='', encoding='utf-8') as file:
                                writer = csv.writer(file)
                                writer.writerow(data_row)
                            saved_plates_set.add(final_text)
                            print(f"✅ Confirmed and saved: {final_text}")
                            text_to_display = f"SAVED: {final_text}"
                        else:
                            text_to_display = f"LOGGED: {final_text}"
                    else:
                        count = current_counts[final_text]
                        text_to_display = f"{final_text} ({count}/{STABLE_DETECTION_THRESHOLD})"
                    
                    last_detections.append((box, text_to_display))

                except Exception as e:
                    pass
        
        ## NEW: Calculate and update performance metrics ##
        end_time = time.perf_counter()
        inference_time_ms = (end_time - start_time) * 1000
        process_times.append(inference_time_ms)
        
        # Calculate average over the last few frames for stability
        avg_inf_time = sum(process_times) / len(process_times)
        fps = 1000 / avg_inf_time if avg_inf_time > 0 else 0
        
        # Get system metrics
        cpu_util = current_process.cpu_percent(interval=None)
        mem_usage_mb = current_process.memory_info().rss / (1024 * 1024)
        
        # Get GPU metrics if available
        gpu_util = "N/A"
        power_consumption = "N/A"
        if NVIDIA_GPU_AVAILABLE:
            gpu_util = f"{pynvml.nvmlDeviceGetUtilizationRates(GPU_HANDLE).gpu}%"
            power_consumption = f"{pynvml.nvmlDeviceGetPowerUsage(GPU_HANDLE) / 1000.0:.2f} W"

        performance_metrics = {
            "Inference Time": f"{avg_inf_time:.2f} ms",
            "Frames Per Second (FPS)": f"{fps:.2f}",
            "CPU Utilization": f"{cpu_util:.2f}%",
            "System Memory Usage": f"{mem_usage_mb:.2f} MB",
            "GPU Utilization": gpu_util,
            "Power Consumption": power_consumption
        }

    # --- DRAWING LOGIC (RUNS EVERY FRAME) ---
    for box, text in last_detections:
        x1, y1, x2, y2 = box
        cv2.rectangle(frame_resized, (x1, y1), (x2, y2), BOX_COLOR, FONT_THICKNESS)
        (text_width, text_height), baseline = cv2.getTextSize(text, FONT, FONT_SCALE, FONT_THICKNESS)
        text_origin = (x1, y1 - 10)
        cv2.rectangle(frame_resized,
                      (text_origin[0], text_origin[1] - text_height - baseline),
                      (text_origin[0] + text_width, text_origin[1] + baseline),
                      TEXT_BG_COLOR, -1)
        cv2.putText(frame_resized, text, text_origin, FONT, FONT_SCALE, TEXT_COLOR, FONT_THICKNESS)

    ## NEW: Draw the performance metrics on the frame ##
    if performance_metrics:
        frame_resized = draw_performance_metrics(frame_resized, performance_metrics)

    cv2.imshow('Live License Plate Detection (Press "q" to exit)', frame_resized)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- CLEANUP ---
cap.release()
cv2.destroyAllWindows()
## NEW: Shutdown NVML if it was initialized ##
if NVIDIA_GPU_AVAILABLE:
    pynvml.nvmlShutdown()
print("Execution finished.")