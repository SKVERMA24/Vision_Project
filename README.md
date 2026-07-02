# Vision ADAS: Advanced Driver Assistance System

A vision-based Advanced Driver Assistance System (ADAS) implemented in Python using classical computer vision techniques for lane tracking and YOLOv8 for vehicle tracking and collision safety analytics.

## Features

1. **Lane Detection and Area Estimation**:
   - Color masking in HSV color space to isolate white and yellow lanes under changing illumination conditions.
   - Sliding window histogram peak detection and 2nd-order polynomial curve fitting ($x = Ay^2 + By + C$).
   - Perspective transform to generate a Bird's-Eye View (BEV) and calculation of drivable lane area.
2. **Vehicle Detection with YOLOv8**:
   - Pretrained YOLOv8 nano model for high-speed object detection filtering for road vehicles (cars, trucks, buses, motorcycles).
3. **Region of Interest (ROI) Masking**:
   - A parameterized trapezoidal road zone directly ahead of the vehicle is used to filter vehicle interactions for counting and forward collision analysis.
4. **Unique Vehicle Tracking and Counting**:
   - A lightweight greedy Intersection over Union (IoU) tracker keeps identity persistence across frames to count unique vehicles entering the ROI exactly once.
5. **Forward Collision Warning (FCW)**:
   - Dynamic collision warning triggers if a tracked vehicle in the ROI gets too close (vertical position threshold), is excessively large (width/area threshold), or is approaching rapidly (rate of box expansion).
6. **Lane Departure Warning (LDW)**:
   - Computes lateral deviation between the vehicle center (camera center) and the lane center. Alerts when offset exceeds $0.35$ meters.
7. **Unified HUD Visualization**:
   - A premium, translucent HUD showing live stats: deviation, vehicle count, collision risk status, and lane positioning. Large flashing alert banners appear during warning states. Text, lines, and boxes dynamically rescale to fit the resolution.

---

## File Structure

All project files are located in a single flat directory:

```
Vision Drive/
├── download_video.py        # Test video downloader helper script
├── requirements.txt         # Project dependencies
├── README.md                # System documentation
├── lane_detector.py         # Preprocessing, BEV warp, polyfit, and offset warnings
├── tracker.py               # IoU-based bounding box tracker
├── vehicle_detector.py      # YOLOv8 inference, ROI checks, and FCW
├── visualizer.py            # HUD drawing and alert overlays
├── main.py                  # Application entrypoint
├── project_video.mp4        # Input test video
├── output_adas.mp4          # Annotated output video file (640x360 optimized)
└── yolov8n.pt               # YOLOv8 model weights
```

---

## Installation & Setup

1. **Install Requirements**:
   Ensure you have Python 3.10+ installed. Install the package dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. **Download Test Video**:
   Download a sample highway driving video to test the ADAS system:
   ```bash
   python download_video.py
   ```

3. **Run the ADAS Pipeline**:
   Process the downloaded video at an optimized $640 \times 360$ resolution to reduce the output video size and double the speed:
   ```bash
   python -B main.py --input project_video.mp4 --output output_adas.mp4 --width 640 --height 360
   ```

   To run at full $1280 \times 720$ resolution:
   ```bash
   python -B main.py --input project_video.mp4 --output output_adas.mp4 --width 1280 --height 720
   ```

   To see a live preview during processing, add the `--show` flag:
   ```bash
   python -B main.py --input project_video.mp4 --output output_adas.mp4 --show
   ```

---

## Key Configurations

You can customize the detection parameters directly inside the source files:
- **ROI Shape**: Modify `self.roi_polygon` in `vehicle_detector.py` to change the safety monitoring zone.
- **Warp Matrix**: Adjust `src_points` and `dst_points` in `lane_detector.py` to match different camera heights/mounting angles.
- **YOLO Confidence**: Change the `conf_threshold` parameter in `main.py` when calling `vehicle_detector.detect` to filter out weak detections.
- **Warning Thresholds**: Modify the deviation threshold (currently `0.35`m) in `lane_detector.py` or collision thresholds (bounding box width and y-coordinate) in `vehicle_detector.py`.
