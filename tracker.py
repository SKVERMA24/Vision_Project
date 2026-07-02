import numpy as np

def calculate_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interWidth = max(0, xB - xA)
    interHeight = max(0, yB - yA)
    interArea = interWidth * interHeight

    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    unionArea = float(boxAArea + boxBArea - interArea)
    if unionArea == 0:
        return 0.0
    return interArea / unionArea

class Track:
    def __init__(self, track_id, box, class_id, conf):
        self.track_id = track_id
        self.box = box  # [x1, y1, x2, y2]
        self.class_id = class_id
        self.conf = conf
        self.age = 0
        self.counted = False
        self.centroid_history = [self.get_centroid()]

    def get_centroid(self):
        x1, y1, x2, y2 = self.box
        return (int((x1 + x2) / 2), int((y1 + y2) / 2))

    def get_bottom_center(self):
        x1, y1, x2, y2 = self.box
        return (int((x1 + x2) / 2), int(y2))

    def update(self, box, conf):
        self.box = box
        self.conf = conf
        self.age = 0
        self.centroid_history.append(self.get_centroid())
        if len(self.centroid_history) > 30:
            self.centroid_history.pop(0)

class IoUTracker:
    def __init__(self, iou_threshold=0.3, max_lost_frames=10):
        self.iou_threshold = iou_threshold
        self.max_lost_frames = max_lost_frames
        self.next_track_id = 1
        self.tracks = []

    def update(self, detections):
        updated_tracks = []
        matched_detections = set()
        matched_tracks = set()

        iou_matrix = []
        for track in self.tracks:
            row = []
            for det in detections:
                row.append(calculate_iou(track.box, det['box']))
            iou_matrix.append(row)

        if len(self.tracks) > 0 and len(detections) > 0:
            flat_matches = []
            for t_idx in range(len(self.tracks)):
                for d_idx in range(len(detections)):
                    iou = iou_matrix[t_idx][d_idx]
                    if iou >= self.iou_threshold:
                        flat_matches.append((iou, t_idx, d_idx))
            
            flat_matches.sort(key=lambda x: x[0], reverse=True)

            for iou, t_idx, d_idx in flat_matches:
                if t_idx not in matched_tracks and d_idx not in matched_detections:
                    if self.tracks[t_idx].class_id == detections[d_idx]['class_id']:
                        self.tracks[t_idx].update(detections[d_idx]['box'], detections[d_idx]['conf'])
                        matched_tracks.add(t_idx)
                        matched_detections.add(d_idx)

        for d_idx, det in enumerate(detections):
            if d_idx not in matched_detections:
                new_track = Track(self.next_track_id, det['box'], det['class_id'], det['conf'])
                self.tracks.append(new_track)
                self.next_track_id += 1

        surviving_tracks = []
        for t_idx, track in enumerate(self.tracks):
            if t_idx not in matched_tracks:
                track.age += 1
            if track.age <= self.max_lost_frames:
                surviving_tracks.append(track)
        
        self.tracks = surviving_tracks
        return self.tracks
