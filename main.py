import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from ultralytics import YOLO
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict, deque
import matplotlib.pyplot as plt
import os
from typing import Dict, List, Tuple, Optional

class PlayerTracker:
    def __init__(self, model_path: str = None):
        """
        Initialize the player tracking system.
        
        Args:
            model_path: Path to custom YOLO model, if None uses YOLOv8 pretrained
        """
        # Load YOLO model - using YOLOv8 as it's excellent for person detection
        if model_path and os.path.exists(model_path):
            self.model = YOLO(model_path)
        else:
            # Use YOLOv8n for speed or YOLOv8x for accuracy
            self.model = YOLO('yolov8n.pt')  # Will download automatically
        
        # Tracking parameters
        self.next_player_id = 1
        self.active_players = {}  # player_id -> PlayerInfo
        self.inactive_players = {}  # player_id -> PlayerInfo (for re-identification)
        self.max_inactive_frames = 30  # Keep inactive players for 30 frames
        self.similarity_threshold = 0.7  # Threshold for re-identification
        self.max_distance_threshold = 100  # Max pixel distance for tracking
        
        # Feature extraction setup
        self.feature_extractor = self._setup_feature_extractor()
        
    def _setup_feature_extractor(self):
        """Setup ResNet feature extractor for player re-identification"""
        from torchvision.models import resnet50
        
        # Load pretrained ResNet50
        model = resnet50(pretrained=True)
        # Remove the final classification layer
        model = torch.nn.Sequential(*list(model.children())[:-1])
        model.eval()
        
        return model
    
    def extract_features(self, image_crop: np.ndarray) -> np.ndarray:
        """Extract features from a player crop for re-identification"""
        if image_crop.size == 0:
            return np.zeros(2048)
        
        # Preprocessing
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        try:
            # Convert BGR to RGB
            image_rgb = cv2.cvtColor(image_crop, cv2.COLOR_BGR2RGB)
            input_tensor = transform(image_rgb).unsqueeze(0)
            
            with torch.no_grad():
                features = self.feature_extractor(input_tensor)
                features = features.squeeze().numpy()
                
            return features
        except Exception as e:
            print(f"Feature extraction error: {e}")
            return np.zeros(2048)
    
    def calculate_similarity(self, features1: np.ndarray, features2: np.ndarray) -> float:
        """Calculate cosine similarity between two feature vectors"""
        if features1.size == 0 or features2.size == 0:
            return 0.0
        
        # Reshape for cosine similarity calculation
        features1 = features1.reshape(1, -1)
        features2 = features2.reshape(1, -1)
        
        similarity = cosine_similarity(features1, features2)[0][0]
        return similarity
    
    def track_players(self, video_path: str, output_path: str = None) -> Dict:
        """
        Main tracking function that processes the entire video.
        
        Args:
            video_path: Path to input video
            output_path: Path to save output video (optional)
            
        Returns:
            Dictionary with tracking results
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        # Video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Setup video writer if output path provided
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        tracking_results = defaultdict(list)
        frame_count = 0
        
        print(f"Processing video: {total_frames} frames at {fps} FPS")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Detect players in current frame
            detections = self.detect_players(frame)
            
            # Track and assign IDs
            tracked_players = self.assign_player_ids(frame, detections, frame_count)
            
            # Update tracking results
            for player_id, bbox, confidence in tracked_players:
                tracking_results[player_id].append({
                    'frame': frame_count,
                    'bbox': bbox,
                    'confidence': confidence
                })
            
            # Draw tracking results on frame
            annotated_frame = self.draw_tracking_results(frame, tracked_players)
            
            if output_path:
                out.write(annotated_frame)
            
            # Update inactive players (for re-identification)
            self.update_inactive_players(frame_count)
            
            if frame_count % 30 == 0:
                print(f"Processed {frame_count}/{total_frames} frames")
        
        cap.release()
        if output_path:
            out.release()
        
        print(f"Tracking completed. Found {len(tracking_results)} unique players.")
        return dict(tracking_results)
    
    def detect_players(self, frame: np.ndarray) -> List[Tuple]:
        """Detect players in the current frame using YOLO"""
        results = self.model(frame, classes=[0])  # Class 0 is 'person'
        
        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = box.conf[0].cpu().numpy()
                    
                    # Filter low confidence detections
                    if confidence > 0.5:
                        detections.append((x1, y1, x2, y2, confidence))
        
        return detections
    
    def assign_player_ids(self, frame: np.ndarray, detections: List[Tuple], frame_count: int) -> List[Tuple]:
        """Assign player IDs to detections, handling re-identification"""
        tracked_players = []
        
        # Extract features for all detections
        detection_features = []
        for x1, y1, x2, y2, conf in detections:
            crop = frame[int(y1):int(y2), int(x1):int(x2)]
            features = self.extract_features(crop)
            detection_features.append(features)
        
        # Match detections with existing active players
        matched_players = set()
        
        for i, (x1, y1, x2, y2, conf) in enumerate(detections):
            bbox = (x1, y1, x2, y2)
            best_match_id = None
            best_similarity = 0
            
            # Try to match with active players first
            for player_id, player_info in self.active_players.items():
                if player_id in matched_players:
                    continue
                
                # Calculate distance from last known position
                last_bbox = player_info['last_bbox']
                center_dist = self.calculate_center_distance(bbox, last_bbox)
                
                if center_dist < self.max_distance_threshold:
                    # Calculate feature similarity
                    similarity = self.calculate_similarity(
                        detection_features[i], 
                        player_info['features']
                    )
                    
                    if similarity > best_similarity and similarity > self.similarity_threshold:
                        best_similarity = similarity
                        best_match_id = player_id
            
            # If no active player match, try inactive players (re-identification)
            if best_match_id is None:
                for player_id, player_info in self.inactive_players.items():
                    similarity = self.calculate_similarity(
                        detection_features[i], 
                        player_info['features']
                    )
                    
                    if similarity > best_similarity and similarity > self.similarity_threshold:
                        best_similarity = similarity
                        best_match_id = player_id
            
            # Assign ID
            if best_match_id is not None:
                # Update existing player
                if best_match_id in self.inactive_players:
                    # Move from inactive to active
                    self.active_players[best_match_id] = self.inactive_players.pop(best_match_id)
                
                self.active_players[best_match_id].update({
                    'last_bbox': bbox,
                    'last_frame': frame_count,
                    'features': detection_features[i]  # Update features
                })
                matched_players.add(best_match_id)
                tracked_players.append((best_match_id, bbox, conf))
            else:
                # Create new player
                new_id = self.next_player_id
                self.next_player_id += 1
                
                self.active_players[new_id] = {
                    'last_bbox': bbox,
                    'last_frame': frame_count,
                    'features': detection_features[i],
                    'first_frame': frame_count
                }
                tracked_players.append((new_id, bbox, conf))
        
        # Move unmatched active players to inactive
        players_to_deactivate = []
        for player_id, player_info in self.active_players.items():
            if player_id not in matched_players:
                # If player hasn't been seen for a while, deactivate
                if frame_count - player_info['last_frame'] > 5:
                    players_to_deactivate.append(player_id)
        
        for player_id in players_to_deactivate:
            self.inactive_players[player_id] = self.active_players.pop(player_id)
        
        return tracked_players
    
    def calculate_center_distance(self, bbox1: Tuple, bbox2: Tuple) -> float:
        """Calculate distance between centers of two bounding boxes"""
        x1_center = (bbox1[0] + bbox1[2]) / 2
        y1_center = (bbox1[1] + bbox1[3]) / 2
        x2_center = (bbox2[0] + bbox2[2]) / 2
        y2_center = (bbox2[1] + bbox2[3]) / 2
        
        distance = np.sqrt((x1_center - x2_center)**2 + (y1_center - y2_center)**2)
        return distance
    
    def update_inactive_players(self, current_frame: int):
        """Remove inactive players that have been inactive for too long"""
        players_to_remove = []
        
        for player_id, player_info in self.inactive_players.items():
            if current_frame - player_info['last_frame'] > self.max_inactive_frames:
                players_to_remove.append(player_id)
        
        for player_id in players_to_remove:
            self.inactive_players.pop(player_id)
    
    def draw_tracking_results(self, frame: np.ndarray, tracked_players: List[Tuple]) -> np.ndarray:
        """Draw bounding boxes and player IDs on the frame"""
        annotated_frame = frame.copy()
        
        # Define colors for different players
        colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
            (255, 0, 255), (0, 255, 255), (128, 0, 128), (255, 165, 0),
            (128, 128, 0), (0, 128, 128), (128, 0, 0), (0, 128, 0)
        ]
        
        for player_id, bbox, confidence in tracked_players:
            x1, y1, x2, y2 = map(int, bbox)
            color = colors[player_id % len(colors)]
            
            # Draw bounding box
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw player ID and confidence
            label = f"Player {player_id} ({confidence:.2f})"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(annotated_frame, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), color, -1)
            cv2.putText(annotated_frame, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        return annotated_frame

def main():
    """Example usage of the PlayerTracker"""
    
    # Initialize tracker
    tracker = PlayerTracker()
    
    # Process video
    video_path = "15sec_input_720p.mp4"
    output_path = "tracked_output.mp4"
    
    try:
        # Track players throughout the video
        results = tracker.track_players(video_path, output_path)
        
        # Print summary
        print("\n=== TRACKING SUMMARY ===")
        for player_id, frames in results.items():
            print(f"Player {player_id}: appeared in {len(frames)} frames")
            print(f"  First appearance: frame {frames[0]['frame']}")
            print(f"  Last appearance: frame {frames[-1]['frame']}")
            
            # Check for re-appearances (gaps in frame sequence)
            frame_numbers = [f['frame'] for f in frames]
            gaps = []
            for i in range(1, len(frame_numbers)):
                if frame_numbers[i] - frame_numbers[i-1] > 1:
                    gaps.append((frame_numbers[i-1], frame_numbers[i]))
            
            if gaps:
                print(f"  Re-appearances detected: {len(gaps)} gaps")
                for start, end in gaps:
                    print(f"    Gap from frame {start} to {end}")
            print()
        
        print(f"Output video saved to: {output_path}")
        
    except Exception as e:
        print(f"Error processing video: {e}")
        print("\nPlease ensure:")
        print("1. The video file '15sec_input_720p.mp4' exists in the current directory")
        print("2. You have installed required packages: pip install ultralytics opencv-python torch torchvision scikit-learn matplotlib")

if __name__ == "__main__":
    main()