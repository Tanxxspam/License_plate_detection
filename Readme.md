## 📊 Project Workflow & Details

### 1. Dataset
- **Size:** 433 pre-annotated vehicle/license plate images.
- **Format:** Originally annotated in Pascal VOC XML format.

### 2. Data Preprocessing
- Developed custom scripts to parse XML bounding box structures (`xmin`, `ymin`, `xmax`, `ymax`).
- Normalized and converted coordinate data into standard YOLO format: `<class_id> <x_center> <y_center> <width> <height>`.

### 3. Model Training
- **Architecture:** YOLOv8 (Object Detection)
- **Epochs:** 100 epochs
- Training was carried out utilizing PyTorch to optimize feature extraction for varying license plate layouts under diverse lighting conditions.

### 4. Performance Metrics
The trained model achieved highly reliable bounding-box accuracy during validation:
- **mAP@0.5:** `0.7` (mean Average Precision at an Intersection over Union threshold of 0.5)
- **mAP@0.5:0.95:** `0.5` (stringent average mAP across multiple IoU thresholds)

### 5. Deployment
- Built and launched a **Streamlit** dashboard enabling users to upload media (images/videos) and view real-time bounding boxes around detected license plates instantly.

---

## ⚙️ Installation & Setup

1. **Clone the Repository:**
   ```bash
   git clone [https://github.com/YOUR_USERNAME/automatic-license-plate-detection.git](https://github.com/Tanxxspam/automatic-license-plate-detection.git)
   cd automatic-license-plate-detection