"""
DriverAid - Expert Rules Engine
Combines Vision (MediaPipe) + ML (CNN) + Rules to make drowsiness decisions.
Implements temporal smoothing and explainable alerts.
"""

import time
from collections import deque
from typing import Optional, Dict, List
import numpy as np


class DrowsinessRule:
    """Base class for drowsiness detection rules."""
    
    def __init__(self, name: str, severity: str):
        """Initialize rule with name and severity (LOW, MEDIUM, HIGH, CRITICAL)."""
        self.name = name
        self.severity = severity
        self.triggered = False
        self.trigger_time = None
        self.trigger_duration = 0.0
    
    def check(self, data: Dict) -> bool:
        """Check if rule is triggered. Must be implemented by subclasses."""
        raise NotImplementedError


class EyeClosureRule(DrowsinessRule):
    """Rule: Eyes closed for extended period."""
    
    def __init__(self, threshold_seconds: float = 2.0, critical_threshold: float = 4.0):
        super().__init__(
            name="Prolonged Eye Closure",
            severity="HIGH"
        )
        self.threshold = threshold_seconds
        self.critical_threshold = critical_threshold
        self.closure_start_time = None
        
    def check(self, data: Dict) -> bool:
        """Check if eyes have been closed too long."""
        current_time = time.time()
        
        cnn_data = data.get('cnn_prediction', {})
        both_closed = cnn_data.get('both_eyes_closed', False)
        
        ear_avg = data.get('vision', {}).get('ear_avg', 0.3)
        ear_closed = ear_avg < 0.18
        
        eyes_closed = both_closed or ear_closed
        
        if eyes_closed:
            if self.closure_start_time is None:
                self.closure_start_time = current_time
            
            duration = current_time - self.closure_start_time
            self.trigger_duration = duration
            
            if duration >= self.critical_threshold:
                self.severity = "CRITICAL"
                self.triggered = True
                self.trigger_time = current_time
                return True
            elif duration >= self.threshold:
                self.severity = "HIGH"
                self.triggered = True
                self.trigger_time = current_time
                return True
        else:
            self.closure_start_time = None
            self.triggered = False
            self.trigger_duration = 0.0
        
        return False


class BlinkRateRule(DrowsinessRule):
    """Rule: Abnormally low blink rate (drowsiness indicator)."""
    
    def __init__(self, window_seconds: float = 60.0, min_blinks: int = 10):
        super().__init__(
            name="Low Blink Rate",
            severity="LOW"
        )
        self.window = window_seconds
        self.min_blinks = min_blinks
        self.blink_timestamps = deque()
        self.last_ear = None
        self.start_time = time.time()
        
    def check(self, data: Dict) -> bool:
        """Detect blinks and check rate."""
        current_time = time.time()
        ear_avg = data.get('vision', {}).get('ear_avg', 0.3)
        
        if self.last_ear is not None:
            if self.last_ear > 0.2 and ear_avg < 0.18:
                self.blink_timestamps.append(current_time)
        
        self.last_ear = ear_avg
        
        while self.blink_timestamps and (current_time - self.blink_timestamps[0]) > self.window:
            self.blink_timestamps.popleft()
        
        time_elapsed = current_time - self.start_time
        if time_elapsed < self.window:
            self.triggered = False
            return False
        
        if len(self.blink_timestamps) < self.min_blinks:
            self.triggered = True
            return True
        
        self.triggered = False
        return False


class HeadPoseRule(DrowsinessRule):
    """Rule: Head tilted/nodding (drowsiness indicator)."""
    
    def __init__(self, pitch_threshold: float = 25.0, duration_seconds: float = 3.0):
        super().__init__(
            name="Head Nodding/Tilting",
            severity="MEDIUM"
        )
        self.pitch_threshold = pitch_threshold
        self.duration_threshold = duration_seconds
        self.tilt_start_time = None
        
    def check(self, data: Dict) -> bool:
        """Check if head is tilted forward/down for too long."""
        current_time = time.time()
        head_pose = data.get('vision', {}).get('head_pose', {})
        pitch = head_pose.get('pitch', 0.0)
        
        is_tilted = pitch > self.pitch_threshold
        
        if is_tilted:
            if self.tilt_start_time is None:
                self.tilt_start_time = current_time
            
            duration = current_time - self.tilt_start_time
            self.trigger_duration = duration
            
            if duration >= self.duration_threshold:
                self.triggered = True
                self.trigger_time = current_time
                return True
        else:
            self.tilt_start_time = None
            self.triggered = False
            self.trigger_duration = 0.0
        
        return False


class ExpertSystem:
    """Combines all signals and makes final alert decisions with temporal smoothing."""
    
    def __init__(self):
        """Initialize expert system with all rules."""
        self.rules = [
            EyeClosureRule(threshold_seconds=2.0),
            HeadPoseRule(pitch_threshold=48.0, duration_seconds=4.0),
            BlinkRateRule(window_seconds=90.0, min_blinks=12),
        ]
        
        self.current_alert_level = "NONE"
        self.alert_history = deque(maxlen=30)
        
        self.frame_count = 0
        self.total_alerts = 0
        
    def analyze(self, vision_data: Optional[Dict], cnn_prediction: Optional[Dict]) -> Dict:
        """
        Analyze current frame and make alert decision.
        Returns dict with alert_level, confidence, triggered_rules, reason, and recommendations.
        """
        self.frame_count += 1
        
        if vision_data is None or not vision_data.get('face_detected', False):
            return {
                'alert_level': "NONE",
                'confidence': 0.0,
                'triggered_rules': [],
                'reason': "No face detected",
                'recommendations': ["Position yourself in front of the camera"],
                'face_detected': False
            }
        
        combined_data = {
            'vision': vision_data,
            'cnn_prediction': cnn_prediction if cnn_prediction else {}
        }
        
        triggered_rules = []
        max_severity = "NONE"
        
        for rule in self.rules:
            if rule.check(combined_data):
                triggered_rules.append({
                    'name': rule.name,
                    'severity': rule.severity,
                    'duration': getattr(rule, 'trigger_duration', 0.0)
                })
                
                if self._severity_level(rule.severity) > self._severity_level(max_severity):
                    max_severity = rule.severity
        
        confidence = min(len(triggered_rules) * 0.3 + 0.4, 1.0) if triggered_rules else 0.0
        
        self.alert_history.append(max_severity)
        smoothed_level = self._smooth_alert_level()
        
        reason = self._generate_reason(triggered_rules, vision_data, cnn_prediction)
        recommendations = self._generate_recommendations(triggered_rules)
        
        if smoothed_level != "NONE":
            self.total_alerts += 1
        
        self.current_alert_level = smoothed_level
        
        return {
            'alert_level': smoothed_level,
            'confidence': confidence,
            'triggered_rules': triggered_rules,
            'reason': reason,
            'recommendations': recommendations,
            'face_detected': True,
            'frame_count': self.frame_count
        }
    
    def _severity_level(self, severity: str) -> int:
        """Convert severity string to numeric level."""
        levels = {"NONE": 0, "LOW": 1, "MEDIUM": 2, "HIGH": 3, "CRITICAL": 4}
        return levels.get(severity, 0)
    
    def _smooth_alert_level(self) -> str:
        """Apply temporal smoothing to prevent alert flicker."""
        if not self.alert_history:
            return "NONE"
        
        level_counts = {"NONE": 0, "LOW": 0, "MEDIUM": 0, "HIGH": 0, "CRITICAL": 0}
        for level in self.alert_history:
            level_counts[level] += 1
        
        if level_counts["CRITICAL"] / len(self.alert_history) > 0.3:
            return "CRITICAL"
        
        if level_counts["HIGH"] / len(self.alert_history) > 0.4:
            return "HIGH"
        
        if level_counts["MEDIUM"] / len(self.alert_history) > 0.5:
            return "MEDIUM"
        
        return "NONE"
    
    def _generate_reason(self, triggered_rules: List[Dict], 
                        vision_data: Dict, cnn_prediction: Dict) -> str:
        """Generate human-readable explanation for alert."""
        if not triggered_rules:
            return "Driver appears alert"
        
        reasons = []
        
        for rule in triggered_rules:
            name = rule['name']
            duration = rule.get('duration', 0.0)
            
            if name == "Prolonged Eye Closure":
                reasons.append(f"Eyes closed for {duration:.1f}s")
            elif name == "Head Nodding/Tilting":
                reasons.append(f"Head tilted for {duration:.1f}s")
            elif name == "Low Blink Rate":
                reasons.append("Abnormally low blink rate")
        
        ear_avg = vision_data.get('ear_avg', 0.0)
        if ear_avg < 0.2:
            reasons.append(f"Low EAR: {ear_avg:.3f}")
        
        return " | ".join(reasons) if reasons else "Multiple drowsiness indicators"
    
    def _generate_recommendations(self, triggered_rules: List[Dict]) -> List[str]:
        """Generate actionable recommendations based on triggered rules."""
        if not triggered_rules:
            return ["Stay alert", "Take breaks every 2 hours"]
        
        recommendations = []
        
        for rule in triggered_rules:
            severity = rule['severity']
            
            if severity == "CRITICAL":
                recommendations.append("PULL OVER IMMEDIATELY")
                recommendations.append("Take a 15-20 minute break")
            elif severity == "HIGH":
                recommendations.append("Find a safe place to rest soon")
                recommendations.append("Open windows for fresh air")
            elif severity == "MEDIUM":
                recommendations.append("Stay vigilant")
                recommendations.append("Consider taking a break soon")
        
        seen = set()
        unique_recommendations = []
        for rec in recommendations:
            if rec not in seen:
                seen.add(rec)
                unique_recommendations.append(rec)
        
        return unique_recommendations[:3]
    
    def get_statistics(self) -> Dict:
        """Get system statistics."""
        return {
            'total_frames': self.frame_count,
            'total_alerts': self.total_alerts,
            'alert_rate': (self.total_alerts / self.frame_count * 100) if self.frame_count > 0 else 0.0,
            'current_alert': self.current_alert_level
        }
    
    def reset(self):
        """Reset the expert system state."""
        self.alert_history.clear()
        self.current_alert_level = "NONE"
        for rule in self.rules:
            rule.triggered = False
            rule.trigger_time = None


def test_expert_system():
    """Test the expert system with simulated data."""
    print("=" * 60)
    print("TESTING EXPERT SYSTEM")
    print("=" * 60)
    
    expert = ExpertSystem()
    
    print("\nTest 1: Normal state (eyes open, good posture)")
    vision_data = {
        'face_detected': True,
        'ear_avg': 0.28,
        'head_pose': {'pitch': 5.0, 'yaw': 0.0, 'roll': 0.0}
    }
    cnn_data = {
        'both_eyes_closed': False,
        'avg_closed_prob': 0.1
    }
    result = expert.analyze(vision_data, cnn_data)
    print(f"   Alert Level: {result['alert_level']}")
    print(f"   Reason: {result['reason']}")
    
    print("\nTest 2: Eyes closed for extended period")
    vision_data = {
        'face_detected': True,
        'ear_avg': 0.12,
        'head_pose': {'pitch': 5.0, 'yaw': 0.0, 'roll': 0.0}
    }
    cnn_data = {
        'both_eyes_closed': True,
        'avg_closed_prob': 0.95
    }
    
    expert.reset()
    
    import time as time_module
    alert_triggered = False
    for i in range(70):
        result = expert.analyze(vision_data, cnn_data)
        time_module.sleep(0.065)
        if result['alert_level'] == "CRITICAL" and not alert_triggered:
            print(f"   CRITICAL alert triggered at frame {i+1} (~{(i+1)*0.065:.1f}s)")
            print(f"   Reason: {result['reason']}")
            print(f"   Recommendations: {result['recommendations'][:2]}")
            alert_triggered = True
    
    if not alert_triggered:
        print(f"   Alert level: {result['alert_level']} (Expected: CRITICAL)")
        print(f"   Debug: Temporal smoothing needs consistent signal over 30 frames")
    
    print("\nTest 3: Head tilted forward (nodding)")
    vision_data = {
        'face_detected': True,
        'ear_avg': 0.25,
        'head_pose': {'pitch': 30.0, 'yaw': 0.0, 'roll': 0.0}
    }
    cnn_data = {
        'both_eyes_closed': False,
        'avg_closed_prob': 0.2
    }
    
    expert.reset()
    alert_triggered = False
    for i in range(85):
        result = expert.analyze(vision_data, cnn_data)
        time_module.sleep(0.065)
        if result['alert_level'] in ["HIGH", "CRITICAL"] and not alert_triggered:
            print(f"   {result['alert_level']} alert triggered at frame {i+1} (~{(i+1)*0.065:.1f}s)")
            print(f"   Reason: {result['reason']}")
            print(f"   Recommendations: {result['recommendations'][:2]}")
            alert_triggered = True
    
    if not alert_triggered:
        print(f"   Alert level: {result['alert_level']} (Expected: HIGH/CRITICAL)")
        print(f"   Debug: Temporal smoothing needs consistent signal over 30 frames")
    
    print("\nSystem Statistics:")
    stats = expert.get_statistics()
    for key, value in stats.items():
        print(f"   {key}: {value}")
    
    print("\n" + "=" * 60)
    print("EXPERT SYSTEM TESTS COMPLETE!")
    print("=" * 60)


if __name__ == '__main__':
    test_expert_system()
