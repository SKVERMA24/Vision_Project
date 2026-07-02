import cv2
import time
import argparse
import os
import sys

from lane_detector import LaneDetector
from tracker import IoUTracker
from vehicle_detector import VehicleDetector
from visualizer import ADASVisualizer

def run_adas_pipeline(input_video_path, output_video_path, target_width=640, target_height=360, show_preview=False):
    if not os.path.exists(input_video_path):
        print(f"Error: Input video '{input_video_path}' not found!")
        sys.exit(1)
        
    print(f"Opening input video: {input_video_path}")
    cap = cv2.VideoCapture(input_video_path)
    
    # Get original properties
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if orig_w == 0 or orig_h == 0:
        print("Error: Could not read video properties. The file might be corrupted.")
        sys.exit(1)
        
    print(f"Original Video: {orig_w}x{orig_h} | Target Processing Size: {target_width}x{target_height}")
    print(f"FPS: {fps:.2f} | Total Frames: {total_frames}")
    
    # Initialize ADAS Modules with target size
    print("Initializing ADAS system modules...")
    lane_detector = LaneDetector(img_size=(target_width, target_height))
    vehicle_detector = VehicleDetector(model_name="yolov8n.pt", img_size=(target_width, target_height))
    tracker = IoUTracker(iou_threshold=0.3, max_lost_frames=8)
    visualizer = ADASVisualizer(img_size=(target_width, target_height))
    
    # Initialize Video Writer
    print(f"Saving output video to: {output_video_path}")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (target_width, target_height))
    
    unique_vehicle_count = 0
    frame_idx = 0
    start_time = time.time()
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_idx += 1
        
        # Resize frame to target processing size
        if frame.shape[1] != target_width or frame.shape[0] != target_height:
            frame = cv2.resize(frame, (target_width, target_height))
            
        # 1. Lane Detection & Position Estimation
        lane_mask, left_fitx, right_fitx, ploty, deviation, departure_warning = lane_detector.process_frame(frame)
        
        # 2. Vehicle Detection (YOLOv8)
        yolo_detections = vehicle_detector.detect(frame, conf_threshold=0.35)
        
        # 3. Vehicle Tracking (IoU Tracker)
        active_tracks = tracker.update(yolo_detections)
        
        # 4. ROI Processing, Counting, and Forward Collision Warning
        unique_vehicle_count, collision_warning = vehicle_detector.process_vehicles(
            active_tracks, unique_vehicle_count
        )
        
        # 5. Draw Visualizations
        annotated_frame = visualizer.draw_overlays(
            frame, lane_mask, vehicle_detector.roi_polygon, active_tracks, 
            unique_vehicle_count, deviation, departure_warning, collision_warning
        )
        
        # Save output frame
        out.write(annotated_frame)
        
        # Live Preview (if requested)
        if show_preview:
            cv2.imshow("ADAS Pipeline Preview", annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Processing interrupted by user.")
                break
                
        # Logging progress
        if frame_idx % 60 == 0 or frame_idx == total_frames:
            elapsed = time.time() - start_time
            current_fps = frame_idx / elapsed
            percent = (frame_idx / total_frames) * 100 if total_frames > 0 else 0
            print(f"Processed frame {frame_idx}/{total_frames} ({percent:.1f}%) | Current Speed: {current_fps:.1f} FPS")
            
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    total_time = time.time() - start_time
    avg_fps = frame_idx / total_time
    print("\nProcessing Complete!")
    print(f"Total processed frames: {frame_idx}")
    print(f"Total time elapsed   : {total_time:.2f} seconds")
    print(f"Average system speed : {avg_fps:.2f} FPS")
    print(f"Final Unique Vehicle Count inside ROI: {unique_vehicle_count}")
    
    return avg_fps, unique_vehicle_count

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Advanced Driver Assistance System (ADAS) Video Pipeline")
    parser.add_argument("--input", type=str, default="project_video.mp4", help="Path to input video file")
    parser.add_argument("--output", type=str, default="output_adas.mp4", help="Path to save annotated video file")
    parser.add_argument("--width", type=int, default=640, help="Target processing width")
    parser.add_argument("--height", type=int, default=360, help="Target processing height")
    parser.add_argument("--show", action="store_true", help="Display live preview during execution")
    
    args = parser.parse_args()
    
    run_adas_pipeline(args.input, args.output, args.width, args.height, args.show)
