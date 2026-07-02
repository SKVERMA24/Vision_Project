import cv2
import numpy as np

class LaneDetector:
    def __init__(self, img_size=(640, 360)):
        self.img_size = img_size  # (width, height)
        w, h = img_size
        
        # Define perspective transform source and destination points
        # These are calibrated for standard front-facing driving cameras
        src_points = np.float32([
            [w * 0.45, h * 0.63],  # top-left
            [w * 0.55, h * 0.63],  # top-right
            [w * 0.85, h * 0.94],  # bottom-right
            [w * 0.18, h * 0.94]   # bottom-left
        ])
        
        dst_points = np.float32([
            [w * 0.20, 0],         # top-left
            [w * 0.80, 0],         # top-right
            [w * 0.80, h],         # bottom-right
            [w * 0.20, h]          # bottom-left
        ])
        
        self.M = cv2.getPerspectiveTransform(src_points, dst_points)
        self.Minv = cv2.getPerspectiveTransform(dst_points, src_points)
        
        # Track lane line polynomial coefficients (2nd order: Ax^2 + Bx + C)
        self.left_fit = None
        self.right_fit = None
        self.detected = False
        self.lost_counter = 0
        self.max_lost = 5
        
        # Parameters for sliding window
        self.nwindows = 9
        self.margin = int(w * 0.06)  # Scale margin with resolution
        self.minpix = int(h * 0.07)  # Scale minimum pixels with resolution
        
        # Metric conversion factors (meters per pixel)
        self.ym_per_pix = 30.0 / h
        self.xm_per_pix = 3.7 / (w * 0.6)

    def threshold_image(self, img):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # Yellow lane mask
        lower_yellow = np.array([15, 60, 80], dtype=np.uint8)
        upper_yellow = np.array([35, 255, 255], dtype=np.uint8)
        yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
        
        # White lane mask (adaptive thresholding)
        v_channel = hsv[:, :, 2]
        avg_brightness = np.mean(v_channel)
        white_val_min = max(180, int(avg_brightness + 50))
        lower_white = np.array([0, 0, min(white_val_min, 220)], dtype=np.uint8)
        upper_white = np.array([180, 45, 255], dtype=np.uint8)
        white_mask = cv2.inRange(hsv, lower_white, upper_white)
        
        color_binary = cv2.bitwise_or(yellow_mask, white_mask)
        
        hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
        l_channel = hls[:, :, 1]
        
        sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0, ksize=3)
        abs_sobelx = np.absolute(sobelx)
        max_sobel = np.max(abs_sobelx) if np.max(abs_sobelx) > 0 else 1
        scaled_sobel = np.uint8(255 * abs_sobelx / max_sobel)
        
        sobel_binary = np.zeros_like(scaled_sobel)
        sobel_binary[(scaled_sobel >= 25) & (scaled_sobel <= 110)] = 255
        
        combined_binary = np.zeros_like(sobel_binary)
        combined_binary[(color_binary > 0) | (sobel_binary > 0)] = 1
        
        return combined_binary

    def warp(self, img):
        return cv2.warpPerspective(img, self.M, self.img_size, flags=cv2.INTER_LINEAR)

    def unwarp(self, img):
        return cv2.warpPerspective(img, self.Minv, self.img_size, flags=cv2.INTER_LINEAR)

    def find_lanes_sliding_window(self, binary_warped):
        histogram = np.sum(binary_warped[binary_warped.shape[0]//2:, :], axis=0)
        midpoint = int(histogram.shape[0] // 2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint
        
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        
        leftx_current = leftx_base
        rightx_current = rightx_base
        window_height = int(binary_warped.shape[0] // self.nwindows)
        
        left_lane_inds = []
        right_lane_inds = []
        
        for window in range(self.nwindows):
            win_y_low = binary_warped.shape[0] - (window + 1) * window_height
            win_y_high = binary_warped.shape[0] - window * window_height
            win_xleft_low = leftx_current - self.margin
            win_xleft_high = leftx_current + self.margin
            win_xright_low = rightx_current - self.margin
            win_xright_high = rightx_current + self.margin
            
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
                              (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
                               (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
            
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)
            
            if len(good_left_inds) > self.minpix:
                leftx_current = int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > self.minpix:
                rightx_current = int(np.mean(nonzerox[good_right_inds]))
                
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
        
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]
        
        if len(leftx) > self.minpix * 2 and len(rightx) > self.minpix * 2:
            self.left_fit = np.polyfit(lefty, leftx, 2)
            self.right_fit = np.polyfit(righty, rightx, 2)
            self.detected = True
            self.lost_counter = 0
            return True
        return False

    def find_lanes_prior(self, binary_warped):
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        
        left_lane_inds = ((nonzerox > (self.left_fit[0] * (nonzeroy**2) + self.left_fit[1] * nonzeroy + self.left_fit[2] - self.margin)) & 
                          (nonzerox < (self.left_fit[0] * (nonzeroy**2) + self.left_fit[1] * nonzeroy + self.left_fit[2] + self.margin)))
        
        right_lane_inds = ((nonzerox > (self.right_fit[0] * (nonzeroy**2) + self.right_fit[1] * nonzeroy + self.right_fit[2] - self.margin)) & 
                           (nonzerox < (self.right_fit[0] * (nonzeroy**2) + self.right_fit[1] * nonzeroy + self.right_fit[2] + self.margin)))
        
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]
        
        if len(leftx) > self.minpix * 2 and len(rightx) > self.minpix * 2:
            left_fit = np.polyfit(lefty, leftx, 2)
            right_fit = np.polyfit(righty, rightx, 2)
            
            h = binary_warped.shape[0]
            left_bottom = left_fit[0] * (h**2) + left_fit[1] * h + left_fit[2]
            right_bottom = right_fit[0] * (h**2) + right_fit[1] * h + right_fit[2]
            lane_width = right_bottom - left_bottom
            
            # Scaled lane width verification limits (roughly 60% of width)
            min_w = self.img_size[0] * 0.35
            max_w = self.img_size[0] * 0.85
            if min_w < lane_width < max_w:
                self.left_fit = left_fit
                self.right_fit = right_fit
                self.lost_counter = 0
                return True
                
        self.lost_counter += 1
        if self.lost_counter > self.max_lost:
            self.detected = False
        return False

    def process_frame(self, frame):
        h, w = frame.shape[:2]
        
        binary = self.threshold_image(frame)
        warped = self.warp(binary)
        
        success = False
        if self.detected and self.left_fit is not None and self.right_fit is not None:
            success = self.find_lanes_prior(warped)
        
        if not success:
            success = self.find_lanes_sliding_window(warped)
            
        if not success or self.left_fit is None or self.right_fit is None:
            return None, None, None, None, 0.0, None

        ploty = np.linspace(0, h - 1, h)
        left_fitx = self.left_fit[0] * (ploty**2) + self.left_fit[1] * ploty + self.left_fit[2]
        right_fitx = self.right_fit[0] * (ploty**2) + self.right_fit[1] * ploty + self.right_fit[2]
        
        left_bottom = left_fitx[-1]
        right_bottom = right_fitx[-1]
        lane_center = (left_bottom + right_bottom) / 2.0
        vehicle_center = w / 2.0
        
        deviation_pixels = vehicle_center - lane_center
        deviation_meters = deviation_pixels * self.xm_per_pix
        
        departure_warning = None
        if deviation_meters > 0.35:
            departure_warning = 'RIGHT'
        elif deviation_meters < -0.35:
            departure_warning = 'LEFT'
            
        warp_zero = np.zeros_like(warped).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
        
        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        pts = np.hstack((pts_left, pts_right))
        
        cv2.fillPoly(color_warp, np.int32([pts]), (0, 255, 0))
        cv2.polylines(color_warp, np.int32([pts_left]), False, (255, 0, 0), int(w * 0.012))
        cv2.polylines(color_warp, np.int32([pts_right]), False, (0, 0, 255), int(w * 0.012))
        
        unwarped_mask = self.unwarp(color_warp)
        
        return unwarped_mask, left_fitx, right_fitx, ploty, deviation_meters, departure_warning
