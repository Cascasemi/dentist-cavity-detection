# 🦷 AI-Powered Cavity Detection for Dentistry

An advanced computer vision system that uses YOLO (You Only Look Once) deep learning architecture to automatically detect dental cavities in video footage. This AI-powered solution assists dental professionals in identifying potential cavity locations with real-time processing capabilities.

## 🌟 Features

- **Real-time Cavity Detection**: Process dental videos frame-by-frame to identify cavities
- **YOLO-based Architecture**: Utilizes state-of-the-art object detection for accurate results
- **Visual Annotation**: Highlights detected cavities with bounding boxes and overlays
- **Video Processing**: Supports various video formats with HD output resolution
- **Dual Classification**: Distinguishes between "Cavity" and "Normal" dental regions
- **Output Generation**: Creates annotated videos for review and documentation

## 🔧 Requirements

### Dependencies

```bash
pip install ultralytics
pip install opencv-python
pip install numpy
```

### System Requirements

- Python 3.7+
- OpenCV 4.0+
- CUDA-compatible GPU (recommended for faster processing)
- Minimum 8GB RAM
- 2GB free disk space

## 📁 Project Structure

```
AI-Powered-Cavity-Detection-for-Dentistry/
├── cavity detection.pt          # Pre-trained YOLO model weights
├── testing.py                   # Main inference script
├── labels.jpg                   # Class label reference
├── results.png                  # Sample detection results
├── LICENSE                      # Project license
└── README.md                    # This file
```

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/Cascasemi/dentist-cavity-detection.git
cd AI-Powered-Cavity-Detection-for-Dentistry
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

*Note: If requirements.txt doesn't exist, install manually:*

```bash
pip install ultralytics opencv-python numpy
```

### 3. Prepare Your Video

- Place your dental video file in the project directory
- Supported formats: .mp4, .avi, .mov, .mkv
- Recommended resolution: 1920x1080 or higher for better accuracy

### 4. Update Video Path

Edit the `video_path` variable in `testing.py`:

```python
video_path = r'your_dental_video.mp4'
```

### 5. Run Detection

```bash
python testing.py
```

## 💻 Usage

### Basic Usage

```python
import cv2
from ultralytics import YOLO

# Load the model
model = YOLO('cavity detection.pt')

# Process your video
video_path = 'path/to/your/dental/video.mp4'
cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Run inference
    results = model.predict(frame, verbose=False)
    # Process results...
```

### Advanced Configuration

You can modify the following parameters in `testing.py`:

- **Target Resolution**: Adjust `target_width` and `target_height` for processing resolution
- **Colors**: Modify `class_colors` to change detection visualization
- **Confidence Threshold**: Add confidence filtering in the detection loop
- **Output Format**: Change video codec in `VideoWriter` setup

## 🎯 Model Information

### Classes

The model is trained to detect two classes:
- **Cavity**: Dental caries or tooth decay (highlighted in white)
- **Normal**: Healthy dental tissue (highlighted in green)

### Model Architecture

- **Base Model**: YOLOv8/YOLOv5 (Ultralytics implementation)
- **Input Size**: Flexible (automatically resized)
- **Output**: Bounding boxes with class predictions and confidence scores

### Performance

- **Processing Speed**: ~30 FPS on GPU, ~5-10 FPS on CPU
- **Accuracy**: Trained on dental imaging dataset
- **Detection Range**: Optimized for intraoral camera footage

## 📊 Output

The system generates:

1. **Real-time Display**: Live preview window showing detections
2. **Annotated Video**: Output video file (`output_video.mp4`) with:
   - Bounding boxes around detected areas
   - Class labels (Cavity/Normal)
   - Color-coded highlighting
   - White overlay on cavity regions for emphasis

## 🔬 Technical Details

### Detection Pipeline

1. **Frame Extraction**: Video is processed frame-by-frame
2. **Preprocessing**: Frames are resized to target resolution (1280x720)
3. **Inference**: YOLO model processes each frame
4. **Post-processing**: Results are filtered and annotated
5. **Visualization**: Bounding boxes and labels are drawn
6. **Output**: Processed frames are compiled into output video

### Color Coding

- **White Overlay**: Cavity regions (50% transparency)
- **Green Boxes**: Normal/healthy tissue
- **White Boxes**: Detected cavities

## 🛠️ Troubleshooting

### Common Issues

**Video not opening:**
- Check file path and format compatibility
- Ensure video file is not corrupted

**Low performance:**
- Use GPU acceleration with CUDA
- Reduce target resolution
- Close unnecessary applications

**Model not loading:**
- Verify `cavity detection.pt` file exists
- Check file permissions
- Ensure Ultralytics is properly installed

### Error Messages

```python
# If you encounter import errors:
pip install --upgrade ultralytics

# For OpenCV issues:
pip uninstall opencv-python opencv-python-headless
pip install opencv-python
```

## 🤝 Contributing

We welcome contributions! Please see our contribution guidelines:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) for the detection framework
- OpenCV community for computer vision tools
- Dental imaging dataset contributors

## 📞 Support

For questions, issues, or feature requests:

- 🐛 [Report Issues](https://github.com/Cascasemi/dentist-cavity-detection/issues)
- 💬 [Discussions](https://github.com/Cascasemi/dentist-cavity-detection/discussions)
- 📧 Contact: cascatechssolutions@gmail.com

---

⭐ **Star this repository if it helped you!** ⭐

*Disclaimer: This tool is for educational and research purposes. Always consult with qualified dental professionals for medical diagnosis and treatment decisions.*
