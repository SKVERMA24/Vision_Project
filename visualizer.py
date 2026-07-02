import cv2
import numpy as np

class ADASVisualizer:
    def __init__(self, img_size=(640, 360)):
        self.img_size = img_size
        self.hud_bg_color = (25, 25, 25)
        self.text_color = (245, 245, 245)
        self.accent_color = (0, 215, 255)
        self.class_names = {2: "Car", 3: "Motorcycle", 5: "Bus", 7: "Truck"}

    def draw_overlays(self, frame, lane_mask, roi_polygon, tracks, unique_count, 
                      deviation, departure_warning, collision_warning):
        annotated_frame = frame.copy()
        w, h = self.img_size
        
        # Determine adaptive scale factors based on image size
        is_small = w <= 640
        font_scale = 0.35 if is_small else 0.5
        font_thickness = 1
        
        # 1. Overlay Lane Area Mask
        if lane_mask is not None:
            annotated_frame = cv2.addWeighted(annotated_frame, 1.0, lane_mask, 0.35, 0.0)
            
        # 2. Overlay ROI Mask (Semi-transparent yellow overlay)
        roi_overlay = annotated_frame.copy()
        cv2.fillPoly(roi_overlay, [roi_polygon], (0, 165, 255))
        cv2.polylines(annotated_frame, [roi_polygon], True, (0, 255, 255), 2, lineType=cv2.LINE_AA)
        cv2.addWeighted(roi_overlay, 0.15, annotated_frame, 0.85, 0, dst=annotated_frame)
        
        # ROI Label
        roi_top_center = (int((roi_polygon[0][0] + roi_polygon[1][0])/2), roi_polygon[0][1] - int(h * 0.015))
        cv2.putText(annotated_frame, "ROI MONITORING ZONE", roi_top_center, 
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.8, (0, 255, 255), font_thickness, cv2.LINE_AA)
        
        # 3. Draw Vehicle Bounding Boxes
        for track in tracks:
            x1, y1, x2, y2 = track.box
            track_id = track.track_id
            label = f"{self.class_names.get(track.class_id, 'Vehicle')} #{track_id}"
            
            bottom_center = track.get_bottom_center()
            is_inside_roi = cv2.pointPolygonTest(roi_polygon, bottom_center, False) >= 0
            
            if is_inside_roi:
                if collision_warning and (y2 > (h * 0.7) or (x2 - x1) > (w * 0.18)):
                    box_color = (0, 0, 255)  # Danger Red
                    label += " [DANGER]"
                else:
                    box_color = (0, 140, 255)  # ROI orange
                    label += " [ROI]"
            else:
                box_color = (0, 255, 0)  # Safe Green
                
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), box_color, 2, cv2.LINE_AA)
            cv2.circle(annotated_frame, bottom_center, 3, box_color, -1)
            
            text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness=1)
            cv2.rectangle(annotated_frame, (x1, y1 - text_size[1] - 8), (x1 + text_size[0] + 6, y1), box_color, -1)
            cv2.putText(annotated_frame, label, (x1 + 3, y1 - 4), 
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1, cv2.LINE_AA)

        # 4. Draw HUD Dashboard Panel
        hud_w = int(w * 0.36) if is_small else 340
        hud_h = int(h * 0.38) if is_small else 185
        hx, hy = int(w * 0.02), int(h * 0.02)
        
        hud_overlay = annotated_frame.copy()
        cv2.rectangle(hud_overlay, (hx, hy), (hx + hud_w, hy + hud_h), self.hud_bg_color, -1)
        cv2.addWeighted(hud_overlay, 0.75, annotated_frame, 0.25, 0, dst=annotated_frame)
        cv2.rectangle(annotated_frame, (hx, hy), (hx + hud_w, hy + hud_h), self.accent_color, 1, cv2.LINE_AA)
        
        # HUD Text spacing
        text_start_y = hy + int(hud_h * 0.18)
        line_height = int(hud_h * 0.15)
        
        # HUD Title
        cv2.putText(annotated_frame, "VISION ADAS v1.0", (hx + 10, text_start_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale * 1.1, self.accent_color, 1 if is_small else 2, cv2.LINE_AA)
        
        cv2.line(annotated_frame, (hx + 10, text_start_y + 6), (hx + hud_w - 10, text_start_y + 6), (80, 80, 80), 1)
        
        unique_str = f"Unique Vehicles: {unique_count}"
        dev_str = f"Center Offset: {abs(deviation):.2f}m ({'Right' if deviation > 0 else 'Left' if deviation < 0 else 'Center'})"
        
        collision_status = "DANGER" if collision_warning else "SAFE"
        collision_color = (0, 0, 255) if collision_warning else (0, 255, 0)
        
        ldw_status = departure_warning if departure_warning else "OK"
        ldw_color = (0, 165, 255) if departure_warning else (0, 255, 0)
        
        # Write rows
        cv2.putText(annotated_frame, unique_str, (hx + 10, text_start_y + int(line_height * 1.5)), 
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, self.text_color, font_thickness, cv2.LINE_AA)
        cv2.putText(annotated_frame, dev_str, (hx + 10, text_start_y + int(line_height * 2.5)), 
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, self.text_color, font_thickness, cv2.LINE_AA)
        
        cv2.putText(annotated_frame, "Collision Risk:", (hx + 10, text_start_y + int(line_height * 3.5)), 
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, self.text_color, font_thickness, cv2.LINE_AA)
        cv2.putText(annotated_frame, collision_status, (hx + int(hud_w * 0.48), text_start_y + int(line_height * 3.5)), 
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, collision_color, 1 if is_small else 2, cv2.LINE_AA)
        
        cv2.putText(annotated_frame, "Lane Position:", (hx + 10, text_start_y + int(line_height * 4.5)), 
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, self.text_color, font_thickness, cv2.LINE_AA)
        cv2.putText(annotated_frame, ldw_status, (hx + int(hud_w * 0.48), text_start_y + int(line_height * 4.5)), 
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, ldw_color, 1 if is_small else 2, cv2.LINE_AA)
        
        # 5. Draw Alert Warning Banners
        if collision_warning:
            banner_w = int(w * 0.70)
            banner_h = int(h * 0.12)
            bx = (w - banner_w) // 2
            by = int(h * 0.04)
            
            banner_overlay = annotated_frame.copy()
            cv2.rectangle(banner_overlay, (bx, by), (bx + banner_w, by + banner_h), (0, 0, 200), -1)
            cv2.addWeighted(banner_overlay, 0.8, annotated_frame, 0.2, 0, dst=annotated_frame)
            cv2.rectangle(annotated_frame, (bx, by), (bx + banner_w, by + banner_h), (0, 0, 255), 2, cv2.LINE_AA)
            
            alert_text = "COLLISION WARNING: BRAKE NOW"
            tx_sz, _ = cv2.getTextSize(alert_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale * 1.2, 2)
            cv2.putText(annotated_frame, alert_text, (bx + (banner_w - tx_sz[0])//2, by + int(banner_h * 0.7)), 
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale * 1.2, (255, 255, 255), 1 if is_small else 2, cv2.LINE_AA)
                        
        elif departure_warning:
            banner_w = int(w * 0.70)
            banner_h = int(h * 0.12)
            bx = (w - banner_w) // 2
            by = int(h * 0.04)
            
            banner_overlay = annotated_frame.copy()
            cv2.rectangle(banner_overlay, (bx, by), (bx + banner_w, by + banner_h), (0, 100, 255), -1)
            cv2.addWeighted(banner_overlay, 0.8, annotated_frame, 0.2, 0, dst=annotated_frame)
            cv2.rectangle(annotated_frame, (bx, by), (bx + banner_w, by + banner_h), (0, 140, 255), 2, cv2.LINE_AA)
            
            alert_text = f"LANE DEPARTURE ALERT: {departure_warning}"
            tx_sz, _ = cv2.getTextSize(alert_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale * 1.2, 2)
            cv2.putText(annotated_frame, alert_text, (bx + (banner_w - tx_sz[0])//2, by + int(banner_h * 0.7)), 
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale * 1.2, (255, 255, 255), 1 if is_small else 2, cv2.LINE_AA)
        
        return annotated_frame
