import cv2
from typing import Tuple

def put_text(
    img,
    text: str,
    position: Tuple[int, int],
    font_scale: float = 0.8,
    color: Tuple[int, int, int] = (255, 255, 255),
    thickness: int = 2,
    font=cv2.FONT_HERSHEY_SIMPLEX,
) -> None:
    """이미지에 텍스트를 그립니다."""
    cv2.putText(img, text, position, font, font_scale, color, thickness)

def draw_line(
    img,
    start_point: Tuple[int, int],
    end_point: Tuple[int, int],
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
) -> None:
    """이미지에 선을 그립니다."""
    cv2.line(img, start_point, end_point, color, thickness)

def draw_circle(
    img,
    center: Tuple[int, int],
    radius: int = 3,
    color: Tuple[int, int, int] = (255, 0, 0),
    thickness: int = -1,
) -> None:
    """이미지에 원을 그립니다."""
    cv2.circle(img, center, radius, color, thickness)