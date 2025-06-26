"""
한글 폰트 렌더링 유틸리티
PIL을 사용한 한글 텍스트 렌더링 기능 제공
"""

from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np

# 전역 폰트 변수
_global_font = None
_font_loaded = False

def get_korean_font(font_size=20):
    """한글 폰트를 한 번만 로드하여 재사용"""
    global _global_font, _font_loaded
    
    if _font_loaded and _global_font:
        return _global_font
    
    # 한글 폰트 로드 (여러 폰트 경로 시도)
    font_paths = [
        "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/System/Library/Fonts/Arial.ttf",  # macOS
        "C:/Windows/Fonts/malgun.ttf",      # Windows
        "/home/woo/kuBig2025/opencv/data/NanumPenScript-Regular.ttf"  # 사용자 지정 경로
    ]
    
    for font_path in font_paths:
        try:
            _global_font = ImageFont.truetype(font_path, font_size)
            _font_loaded = True
            print(f"한글 폰트 로드 성공: {font_path}")
            break
        except Exception as e:
            continue
    
    if not _font_loaded:
        # 폰트 로드 실패 시 기본 폰트 사용
        _global_font = ImageFont.load_default()
        print("기본 폰트 사용")
        _font_loaded = True
    
    return _global_font

def put_korean_text(img, text, position, font_size=20, color=(255, 255, 255)):
    """PIL을 사용한 한글 텍스트 렌더링 함수"""
    try:
        # PIL 이미지로 변환
        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        
        # 폰트 가져오기
        font = get_korean_font(font_size)
        
        # 색상을 RGB로 변환 (PIL은 RGB 사용)
        color_rgb = (color[2], color[1], color[0])  # BGR to RGB
        
        # 텍스트 그리기
        draw.text(position, text, font=font, fill=color_rgb)
        
        # OpenCV 이미지로 변환
        img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        return img_cv
        
    except Exception as e:
        print(f"PIL 한글 텍스트 렌더링 오류: {e}")
        # 오류 시 기본 OpenCV 텍스트 사용
        cv2.putText(img, text, position, cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        return img

def put_text_safe(img, text, position, font_size=0.6, color=(255, 255, 255), thickness=2):
    """안전한 텍스트 표시 함수"""
    try:
        cv2.putText(img, text, position, cv2.FONT_HERSHEY_SIMPLEX, font_size, color, thickness)
    except Exception as e:
        print(f"텍스트 표시 오류: {e}")
    return img

def render_korean_text_on_image(image, text, position, font_size=20, color=(255, 255, 255), background_color=None):
    """이미지에 한글 텍스트를 배경과 함께 렌더링"""
    try:
        # PIL 이미지로 변환
        img_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        
        # 폰트 가져오기
        font = get_korean_font(font_size)
        
        # 색상을 RGB로 변환
        color_rgb = (color[2], color[1], color[0])  # BGR to RGB
        
        # 배경색이 지정된 경우 배경 박스 그리기
        if background_color:
            bg_color_rgb = (background_color[2], background_color[1], background_color[0])
            bbox = draw.textbbox(position, text, font=font)
            draw.rectangle(bbox, fill=bg_color_rgb)
        
        # 텍스트 그리기
        draw.text(position, text, font=font, fill=color_rgb)
        
        # OpenCV 이미지로 변환
        img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        return img_cv
        
    except Exception as e:
        print(f"한글 텍스트 렌더링 오류: {e}")
        # 오류 시 기본 OpenCV 텍스트 사용
        cv2.putText(image, text, position, cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        return image 