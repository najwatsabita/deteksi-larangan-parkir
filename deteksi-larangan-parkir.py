import cv2
import numpy as np
from ultralytics import YOLO
import time
import json
import os

class ParkingDetector:
    def __init__(self, video_path, model_path="bestv8.pt", areas_path='restricted_areas.json', logo_path='logo.png'):
        self.video_path = video_path
        self.model = YOLO(model_path)
        
        # Load video and get properties
        self.cap = cv2.VideoCapture(video_path)
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        
        # Load and scale restricted areas
        self.restricted_areas = self.load_and_scale_areas(areas_path)
        
        # Colors for visualization
        self.vehicle_colors = {
            "car": "#0000C0",
            "motorcycle": "#C00000",
            "bus": "#C000C0",
            "truck": "#00C000",
        }
        
        # Dictionary to store vehicle IDs and their tracking info
        self.vehicle_trackers = {}
        self.next_vehicle_id = 1

        # Load logo
        self.logo = cv2.imread(logo_path, cv2.IMREAD_UNCHANGED)  # Load logo with alpha channel

    def load_and_scale_areas(self, areas_path):
        try:
            with open(areas_path, 'r') as f:
                areas = json.load(f)
                
            screen_width = 1920
            screen_height = 1080
            
            scale_x = self.frame_width / screen_width
            scale_y = self.frame_height / screen_height
            
            for area in areas:
                coords = area['coordinates']
                area['coordinates'] = [
                    int(coords[0] * scale_x),
                    int(coords[1] * scale_y),
                    int(coords[2] * scale_x),
                    int(coords[3] * scale_y)
                ]
            return areas
            
        except FileNotFoundError:
            print("No restricted areas JSON found. Using default areas.")
            return []

    def hex_to_rgb_with_opacity(self, hex_color, opacity=1.0):
        rgb = tuple(int(hex_color[i:i + 2], 16) for i in (1, 3, 5))
        return (rgb[2], rgb[1], rgb[0], int(255 * opacity))

    def is_vehicle_in_restricted_area(self, bbox, area):
        x_min, y_min, x_max, y_max = bbox
        area_x1, area_y1, area_x2, area_y2 = area['coordinates']
        
        center_x = (x_min + x_max) // 2
        center_y = (y_min + y_max) // 2
        
        return (area_x1 <= center_x <= area_x2 and 
                area_y1 <= center_y <= area_y2)

    def get_vehicle_id(self, bbox, vehicle_type):
        center_x = (bbox[0] + bbox[2]) // 2
        center_y = (bbox[1] + bbox[3]) // 2
        
        # Check if this detection matches any existing tracked vehicles
        for vehicle_id, tracker in list(self.vehicle_trackers.items()):
            old_center_x = tracker['center'][0]
            old_center_y = tracker['center'][1]
            
            # Calculate distance between current detection and tracked vehicle
            distance = np.sqrt((center_x - old_center_x)**2 + (center_y - old_center_y)**2)
            
            # If the detection is close to a tracked vehicle and of the same type
            if distance < 100 and vehicle_type == tracker['type']:
                # Update tracker position
                self.vehicle_trackers[vehicle_id] = {
                    'center': (center_x, center_y),
                    'type': vehicle_type,
                    'last_seen': time.time()
                }
                return vehicle_id
        
        # If no match found, create new tracker
        new_id = self.next_vehicle_id
        self.vehicle_trackers[new_id] = {
            'center': (center_x, center_y),
            'type': vehicle_type,
            'last_seen': time.time()
        }
        self.next_vehicle_id += 1
        return new_id

    def clean_old_trackers(self, current_time, threshold=2.0):
        # Remove trackers that haven't been updated recently
        for vehicle_id in list(self.vehicle_trackers.keys()):
            if current_time - self.vehicle_trackers[vehicle_id]['last_seen'] > threshold:
                del self.vehicle_trackers[vehicle_id]

    def overlay_logo(self, frame):
        if self.logo is None:
            return frame

        # Get logo dimensions and scale if necessary
        logo_h, logo_w, logo_channels = self.logo.shape
        scale = 0.1  # Scale factor for logo
        logo_h = int(logo_h * scale)
        logo_w = int(logo_w * scale)

        logo_resized = cv2.resize(self.logo, (logo_w, logo_h), interpolation=cv2.INTER_AREA)

        # Extract alpha channel for blending
        if logo_resized.shape[2] == 4:  # Check if logo has alpha channel
            alpha_channel = logo_resized[:, :, 3] / 255.0
            logo_rgb = logo_resized[:, :, :3]
        else:
            alpha_channel = np.ones((logo_h, logo_w), dtype=np.float32)
            logo_rgb = logo_resized

        # Position logo at bottom-right corner
        x_offset = frame.shape[1] - logo_w - 10
        y_offset = frame.shape[0] - logo_h - 10

        roi = frame[y_offset:y_offset + logo_h, x_offset:x_offset + logo_w]

        # Blend logo with ROI
        for c in range(3):  # Loop over RGB channels
            roi[:, :, c] = (alpha_channel * logo_rgb[:, :, c] + (1 - alpha_channel) * roi[:, :, c])

        frame[y_offset:y_offset + logo_h, x_offset:x_offset + logo_w] = roi
        return frame

    def process_frame(self, frame):
        # Create overlay for restricted areas with new color (#C00000) and 50% opacity
        overlay = frame.copy()
        restricted_area_color = self.hex_to_rgb_with_opacity("#C00000", 0.5)
        for area in self.restricted_areas:
            x1, y1, x2, y2 = area['coordinates']
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 200), -1)
        
        # Combine overlay with frame
        frame = cv2.addWeighted(overlay, 0.5, frame, 0.5, 0)
        
        # Clean old trackers
        self.clean_old_trackers(time.time())
        
        # Detect vehicles
        results = self.model(frame)
        detections = results[0].boxes.xyxy.cpu().numpy()
        classes = results[0].boxes.cls.cpu().numpy()
        
        # Add title with new position and colors
        title_text = "Deteksi Larangan Parkir Kendaraan"
        (text_width, text_height), _ = cv2.getTextSize(title_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(frame, (20, 10), (20 + text_width + 10, 40), (0, 0, 0), -1)
        cv2.putText(frame, title_text, (25, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Process detections
        for det, cls in zip(detections, classes):
            x_min, y_min, x_max, y_max = map(int, det[:4])
            vehicle_type = self.model.names[int(cls)]
            color_hex = self.vehicle_colors.get(vehicle_type, "#FFFFFF")
            color_rgb = self.hex_to_rgb_with_opacity(color_hex)
            
            # Get or assign vehicle ID
            vehicle_id = self.get_vehicle_id((x_min, y_min, x_max, y_max), vehicle_type)
            
            # Check for violations
            violation = any(self.is_vehicle_in_restricted_area(
                (x_min, y_min, x_max, y_max), area) 
                for area in self.restricted_areas)
            
            # Draw bounding box with blinking effect for violations
            if violation:
                blink_color = self.hex_to_rgb_with_opacity("#C00000", 0.8) if int(time.time() * 2) % 2 == 0 else color_rgb
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), blink_color[:3], 2)
                
                # Add warning text for violations
                warning_text = "Kendaraan Ini Masuk Area Terlarang Parkir"
                text_size = cv2.getTextSize(warning_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                text_x = x_min + (x_max - x_min - text_size[0]) // 2
                cv2.putText(frame, warning_text, (text_x, y_min - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 200), 2)
                
            # Draw bounding box and vehicle info
            else:
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color_rgb[:3], 2)
            
            vehicle_text = f"{vehicle_type.upper()} ({vehicle_id})"
            cv2.putText(frame, vehicle_text, (x_min, y_min - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Add logo
        frame = self.overlay_logo(frame)
        
        return frame

    def run(self, output_path):
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, self.fps, (self.frame_width, self.frame_height))
        
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break
            
            processed_frame = self.process_frame(frame)
            out.write(processed_frame)
        
        self.cap.release()
        out.release()

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    video_path = os.path.join(current_dir, "areabaru.mp4")
    output_path = os.path.join(current_dir, "output_video_detection7.mp4")
    model_path = os.path.join(current_dir, "bestv8.pt")
    areas_path = os.path.join(current_dir, "restricted_areas.json")
    logo_path = os.path.join(current_dir, "augenio.png")

    detector = ParkingDetector(
        video_path=video_path,
        model_path=model_path,
        areas_path=areas_path,
        logo_path=logo_path
    )
    
    detector.run(output_path)
