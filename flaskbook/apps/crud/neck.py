import math
import numpy as np

class PostureAnalyzer:
    def __init__(self):
        import mediapipe as mp
        self.mp_pose = mp.solutions.pose

    def calculate_angle(self, a, b, c):
        a_pt = np.array([a.x, a.y]) if not isinstance(a, np.ndarray) else a
        b_pt = np.array([b.x, b.y]) if not isinstance(b, np.ndarray) else b
        c_pt = np.array([c.x, c.y]) if not isinstance(c, np.ndarray) else c

        ba = a_pt - b_pt
        bc = c_pt - b_pt

        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
        return np.degrees(angle)


    def calculate_neck_angle(self, landmarks):
        mp = self.mp_pose
        left_ear = landmarks[mp.PoseLandmark.LEFT_EAR]
        right_ear = landmarks[mp.PoseLandmark.RIGHT_EAR]
        left_shoulder = landmarks[mp.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = landmarks[mp.PoseLandmark.RIGHT_SHOULDER]
        ear_center = np.array([(left_ear.x + right_ear.x) / 2, (left_ear.y + right_ear.y) / 2])
        shoulder_center = np.array([(left_shoulder.x + right_shoulder.x) / 2, (left_shoulder.y + right_shoulder.y) / 2])
        vertical = np.array([shoulder_center[0], ear_center[1]])
        return self.calculate_angle(ear_center, shoulder_center, vertical)

    def grade_neck_posture(self, neck_angle):
        if neck_angle <= 5:
            return 'A', "완벽한 자세"
        elif neck_angle <= 10:
            return 'B', "양호한 자세"
        elif neck_angle <= 15:
            return 'C', "보통 자세"
        else:
            return 'D', "나쁜 자세"

    def analyze_turtle_neck_detailed(self, landmarks):
        mp = self.mp_pose
        neck_angle = self.calculate_neck_angle(landmarks)
        grade, desc = self.grade_neck_posture(neck_angle)
        neck_top = (
            (landmarks[mp.PoseLandmark.LEFT_EAR].x + landmarks[mp.PoseLandmark.RIGHT_EAR].x) / 2,
            (landmarks[mp.PoseLandmark.LEFT_EAR].y + landmarks[mp.PoseLandmark.RIGHT_EAR].y) / 2
        )
        shoulder_center = (
            (landmarks[mp.PoseLandmark.LEFT_SHOULDER].x + landmarks[mp.PoseLandmark.RIGHT_SHOULDER].x) / 2,
            (landmarks[mp.PoseLandmark.LEFT_SHOULDER].y + landmarks[mp.PoseLandmark.RIGHT_SHOULDER].y) / 2
        )
        vertical_deviation = abs(neck_top[0] - shoulder_center[0])
        return {
            'neck_angle': neck_angle,
            'grade': grade,
            'grade_description': desc,
            'vertical_deviation': vertical_deviation,
            'neck_top': neck_top,
            'shoulder_center': shoulder_center
        }

    def analyze_spine_curvature(self, landmarks):
        mp = self.mp_pose
        scx = (landmarks[mp.PoseLandmark.LEFT_SHOULDER].x + landmarks[mp.PoseLandmark.RIGHT_SHOULDER].x) / 2
        scy = (landmarks[mp.PoseLandmark.LEFT_SHOULDER].y + landmarks[mp.PoseLandmark.RIGHT_SHOULDER].y) / 2
        hcx = (landmarks[mp.PoseLandmark.LEFT_HIP].x + landmarks[mp.PoseLandmark.RIGHT_HIP].x) / 2
        hcy = (landmarks[mp.PoseLandmark.LEFT_HIP].y + landmarks[mp.PoseLandmark.RIGHT_HIP].y) / 2
        angle = math.degrees(math.atan2(hcx - scx, hcy - scy))
        return {
            'is_hunched': abs(angle) > 12,
            'spine_angle': angle,
            'shoulder_center': (scx, scy),
            'hip_center': (hcx, hcy),
            'spine_mid': ((scx + hcx) / 2, (scy + hcy) / 2)
        }

    def analyze_shoulder_asymmetry(self, landmarks):
        mp = self.mp_pose
        l = landmarks[mp.PoseLandmark.LEFT_SHOULDER]
        r = landmarks[mp.PoseLandmark.RIGHT_SHOULDER]
        diff = abs(l.y - r.y)
        angle = math.degrees(math.atan2(r.y - l.y, r.x - l.x))
        return {
            'is_asymmetric': diff > 0.02,
            'height_difference': diff,
            'shoulder_angle': angle,
            'higher_shoulder': "왼쪽" if l.y < r.y else "오른쪽",
            'shoulder_mid': ((l.x + r.x) / 2, (l.y + r.y) / 2),
            'left_shoulder': (l.x, l.y),
            'right_shoulder': (r.x, r.y)
        }

    def analyze_pelvic_tilt(self, landmarks):
        mp = self.mp_pose
        l = landmarks[mp.PoseLandmark.LEFT_HIP]
        r = landmarks[mp.PoseLandmark.RIGHT_HIP]
        lk = landmarks[mp.PoseLandmark.LEFT_KNEE]
        rk = landmarks[mp.PoseLandmark.RIGHT_KNEE]
        diff = abs(l.y - r.y)
        angle = math.degrees(math.atan2(r.y - l.y, r.x - l.x))
        return {
            'is_tilted': diff > 0.015,
            'height_difference': diff,
            'pelvic_angle': angle,
            'higher_hip': "왼쪽" if l.y < r.y else "오른쪽",
            'pelvic_center': ((l.x + r.x) / 2, (l.y + r.y) / 2),
            'left_hip': (l.x, l.y),
            'right_hip': (r.x, r.y),
            'left_hip_knee_angle': math.degrees(math.atan2(lk.y - l.y, lk.x - l.x)),
            'right_hip_knee_angle': math.degrees(math.atan2(rk.y - r.y, rk.x - r.x))
        }

    def analyze_spine_twisting(self, landmarks):
        mp = self.mp_pose
        l_s = landmarks[mp.PoseLandmark.LEFT_SHOULDER]
        r_s = landmarks[mp.PoseLandmark.RIGHT_SHOULDER]
        l_h = landmarks[mp.PoseLandmark.LEFT_HIP]
        r_h = landmarks[mp.PoseLandmark.RIGHT_HIP]
        l_e = landmarks[mp.PoseLandmark.LEFT_EAR]
        r_e = landmarks[mp.PoseLandmark.RIGHT_EAR]
        shoulder_center_x = (l_s.x + r_s.x) / 2
        hip_center_x = (l_h.x + r_h.x) / 2
        hip_center_y = (l_h.y + r_h.y) / 2
        alignment = abs(shoulder_center_x - hip_center_x)
        ear_center_x = (l_e.x + r_e.x) / 2
        ear_center_y = (l_e.y + r_e.y) / 2
        side_spine_angle = math.degrees(math.atan2(hip_center_x - ear_center_x, hip_center_y - ear_center_y))
        return {
            'is_twisted': alignment > 0.03,
            'spine_alignment': alignment,
            'side_spine_angle': side_spine_angle,
            'ear_center': (ear_center_x, ear_center_y),
            'shoulder_center': (shoulder_center_x, (l_s.y + r_s.y) / 2),
            'hip_center': (hip_center_x, hip_center_y)
        }
