import cv2
import numpy as np
from ultralytics import YOLO

class VehicleDetector:
    def __init__(self, model_name="yolov8n.pt", img_size=(640, 360)):
        self.img_size = img_size
        w, h = img_size
        
        self.model = YOLO(model_name)
        self.target_classes = {2, 3, 5, 7}  # car, motorcycle, bus, truck
        
        # Define safety Region of Interest (ROI) Trapezoid polygon (scaled)
        self.roi_polygon = np.array([
            [int(w * 0.40), int(h * 0.65)],
            [int(w * 0.60), int(h * 0.65)],
            [int(w * 0.90), int(h * 0.95)],
            [int(w * 0.10), int(h * 0.95)]
        ], dtype=np.int32)
        
        self.prev_track_states = {}

    def get_roi_mask(self):
        mask = np.zeros((self.img_size[1], self.img_size[0]), dtype=np.uint8)
        cv2.fillPoly(mask, [self.roi_polygon], 255)
        return mask

    def detect(self, frame, conf_threshold=0.3):
        results = self.model(frame, verbose=False)[0]
        detections = []
        
        for box in results.boxes:
            cls_id = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            
            if cls_id in self.target_classes and conf >= conf_threshold:
                coords = box.xyxy[0].tolist()
                detections.append({
                    'box': [int(coords[0]), int(coords[1]), int(coords[2]), int(coords[3])],
                    'class_id': cls_id,
                    'conf': conf
                })
        
        return detections

    def process_vehicles(self, tracks, unique_count):
        collision_warning = False
        
        for track in tracks:
            bottom_center = track.get_bottom_center()
            is_inside_roi = cv2.pointPolygonTest(self.roi_polygon, bottom_center, False) >= 0
            
            if is_inside_roi and not track.counted:
                track.counted = True
                unique_count += 1
            
            if is_inside_roi:
                x1, y1, x2, y2 = track.box
                width = x2 - x1
                height = y2 - y1
                area = width * height
                
                # Proximity checks scaled to the current resolution
                if y2 > (self.img_size[1] * 0.82) or width > (self.img_size[0] * 0.22):
                    collision_warning = True
                
                if track.track_id in self.prev_track_states:
                    prev_area, frames_count = self.prev_track_states[track.track_id]
                    if frames_count >= 3:
                        expansion_ratio = area / prev_area
                        if expansion_ratio > 1.15:
                            collision_warning = True
                        self.prev_track_states[track.track_id] = (area, 0)
                    else:
                        self.prev_track_states[track.track_id] = (prev_area, frames_count + 1)
                else:
                    self.prev_track_states[track.track_id] = (area, 0)
        
        active_ids = {t.track_id for t in tracks}
        self.prev_track_states = {k: v for k, v in self.prev_track_states.items() if k in active_ids}
        
        return unique_count, collision_warning
