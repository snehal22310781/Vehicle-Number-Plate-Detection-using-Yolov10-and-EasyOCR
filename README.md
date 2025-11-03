---

# 🚘 Vehicle Number Plate Detection using OpenCV & EasyOCR

## 📘 Project Overview

This project focuses on detecting and recognizing **vehicle number plates** from images or videos using **OpenCV** and **EasyOCR**.
It uses a **YOLO-based object detection model** (or Haar Cascade, depending on implementation) to detect number plates and then applies **Optical Character Recognition (OCR)** to extract the alphanumeric text.

This system can be used in **traffic monitoring**, **toll collection**, and **parking management** applications.

---

## 🧠 Project Workflow

1. **Input:** Image or video stream containing vehicles.
2. **Detection:** Model detects and crops the number plate region.
3. **Recognition:** EasyOCR extracts text from the cropped region.
4. **Output:** Recognized vehicle number displayed on screen and optionally stored in a `.csv` file.

---

## ⚙️ Technologies Used

| Tool / Library      | Purpose                         |
| ------------------- | ------------------------------- |
| Python 3.x          | Main programming language       |
| OpenCV              | Image processing & detection    |
| EasyOCR             | Optical character recognition   |
| YOLO / Haar Cascade | Object detection (number plate) |
| NumPy & Pandas      | Data handling and storage       |
| Matplotlib          | Visualization (optional)        |

---

## 📁 Project Structure

```
/Vehicle-Number-Plate-Detection
│
├── main.py                 # Main detection and recognition script
├── detector/               # YOLO/Haar Cascade model files
│   ├── best.pt / haarcascade_russian_plate_number.xml
├── output/
│   ├── detected_plates/    # Cropped plate images
│   └── results.csv         # OCR results with timestamps
├── dataset.txt             # Dataset link (Kaggle or custom)
├── Project_Link.txt        # GitHub or Kaggle repo link
└── README.md               # Project documentation
```

---

## 🧾 Dataset

If you’re using a public dataset, include the link here:
📂 **Dataset Link:** [Indian Vehicle Number Plate Dataset (Kaggle)](https://www.kaggle.com/datasets/dataclusterlabs/indian-vehicle-dataset)

---

## 🚀 How to Run the Project

### 1️⃣ Install Dependencies

```bash
pip install opencv-python easyocr ultralytics numpy pandas matplotlib
```

### 2️⃣ Run the Script

```bash
python main.py
```

### 3️⃣ Output Example

* The detected number plate region is displayed.
* The recognized text is printed on the console and saved in `results.csv`.

---

## 🧩 Example Output

| Input Image          | Detected Plate       | Extracted Text |
| -------------------- | -------------------- | -------------- |
| ![car](example1.jpg) | ![plate](plate1.jpg) | MH12AB1234     |

---

## 🎯 Applications

* Smart Parking Systems
* Traffic Surveillance
* Toll Booth Automation
* Vehicle Tracking Systems

---

## 🚀 Future Improvements

* Integrate with **Raspberry Pi** for real-time use
* Store data in cloud database or Firebase

---

