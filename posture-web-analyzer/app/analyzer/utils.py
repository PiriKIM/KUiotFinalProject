import numpy as np
import math

def calculate_angle(a, b, c):
    """세 점(a, b, c)으로 각도 계산 (degree)"""
    a = np.array([a.x, a.y])
    b = np.array([b.x, b.y])
    c = np.array([c.x, c.y])
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle

def calculate_distance(point1, point2):
    """두 점(point1, point2) 사이의 거리 계산"""
    return math.sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2)

def get_center_point(point1, point2):
    """두 점의 중간 좌표 반환"""
    return ((point1.x + point2.x) / 2, (point1.y + point2.y) / 2)