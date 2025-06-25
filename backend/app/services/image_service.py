import cv2
import numpy as np
import base64
import io
from PIL import Image, ImageEnhance, ImageFilter
import logging
from typing import Dict, Any, Optional, Tuple, List
import tempfile
import os

logger = logging.getLogger(__name__)

class ImageService:
    """이미지 처리 서비스"""
    
    def __init__(self):
        """이미지 처리 서비스 초기화"""
        self.max_image_size = 10 * 1024 * 1024  # 10MB
        self.supported_formats = ['.jpg', '.jpeg', '.png', '.webp']
        self.quality_settings = {
            'high': {'quality': 95, 'compression': 0},
            'medium': {'quality': 85, 'compression': 5},
            'low': {'quality': 75, 'compression': 10}
        }
        
        logger.info("ImageService 초기화 완료")
    
    def process_image(self, image_data: bytes, processing_options: Dict[str, Any]) -> Dict[str, Any]:
        """
        이미지 처리 메인 함수
        
        Args:
            image_data: 원본 이미지 데이터
            processing_options: 처리 옵션
            
        Returns:
            Dict: 처리된 이미지 정보
        """
        try:
            # 이미지 검증
            validation_result = self._validate_image(image_data)
            if not validation_result['valid']:
                return validation_result
            
            # 이미지 로드
            image = self._load_image(image_data)
            if image is None:
                return {"success": False, "error": "이미지를 로드할 수 없습니다."}
            
            # 처리 옵션 적용
            processed_image = self._apply_processing(image, processing_options)
            
            # 이미지 인코딩
            encoded_image = self._encode_image(processed_image, processing_options.get('quality', 'medium'))
            
            return {
                "success": True,
                "image_data": encoded_image,
                "format": processing_options.get('format', 'jpeg'),
                "size": len(encoded_image),
                "dimensions": processed_image.size
            }
            
        except Exception as e:
            logger.error(f"이미지 처리 중 오류: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def _validate_image(self, image_data: bytes) -> Dict[str, Any]:
        """
        이미지 유효성 검사
        
        Args:
            image_data: 이미지 데이터
            
        Returns:
            Dict: 검증 결과
        """
        # 크기 검사
        if len(image_data) > self.max_image_size:
            return {
                "valid": False,
                "error": f"이미지 크기가 너무 큽니다. 최대 {self.max_image_size // (1024*1024)}MB까지 지원됩니다."
            }
        
        # 형식 검사
        try:
            image = Image.open(io.BytesIO(image_data))
            format_name = image.format.lower()
            
            if format_name not in ['jpeg', 'jpg', 'png', 'webp']:
                return {
                    "valid": False,
                    "error": f"지원하지 않는 이미지 형식입니다: {format_name}"
                }
            
            return {"valid": True, "format": format_name}
            
        except Exception as e:
            return {
                "valid": False,
                "error": f"이미지 형식을 확인할 수 없습니다: {str(e)}"
            }
    
    def _load_image(self, image_data: bytes) -> Optional[Image.Image]:
        """
        이미지 로드
        
        Args:
            image_data: 이미지 데이터
            
        Returns:
            PIL.Image: 로드된 이미지
        """
        try:
            image = Image.open(io.BytesIO(image_data))
            
            # RGB로 변환 (RGBA인 경우)
            if image.mode in ('RGBA', 'LA', 'P'):
                image = image.convert('RGB')
            
            return image
            
        except Exception as e:
            logger.error(f"이미지 로드 오류: {str(e)}")
            return None
    
    def _apply_processing(self, image: Image.Image, options: Dict[str, Any]) -> Image.Image:
        """
        이미지 처리 옵션 적용
        
        Args:
            image: 원본 이미지
            options: 처리 옵션
            
        Returns:
            PIL.Image: 처리된 이미지
        """
        processed_image = image.copy()
        
        # 크기 조정
        if 'resize' in options:
            processed_image = self._resize_image(processed_image, options['resize'])
        
        # 밝기 조정
        if 'brightness' in options:
            processed_image = self._adjust_brightness(processed_image, options['brightness'])
        
        # 대비 조정
        if 'contrast' in options:
            processed_image = self._adjust_contrast(processed_image, options['contrast'])
        
        # 선명도 조정
        if 'sharpness' in options:
            processed_image = self._adjust_sharpness(processed_image, options['sharpness'])
        
        # 노이즈 제거
        if options.get('denoise', False):
            processed_image = self._denoise_image(processed_image)
        
        # 자동 밝기 조정
        if options.get('auto_brightness', False):
            processed_image = self._auto_adjust_brightness(processed_image)
        
        return processed_image
    
    def _resize_image(self, image: Image.Image, resize_options: Dict[str, Any]) -> Image.Image:
        """
        이미지 크기 조정
        
        Args:
            image: 원본 이미지
            resize_options: 크기 조정 옵션
            
        Returns:
            PIL.Image: 크기 조정된 이미지
        """
        width, height = image.size
        
        if 'max_width' in resize_options:
            max_width = resize_options['max_width']
            if width > max_width:
                ratio = max_width / width
                new_width = max_width
                new_height = int(height * ratio)
                image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        if 'max_height' in resize_options:
            max_height = resize_options['max_height']
            if height > max_height:
                ratio = max_height / height
                new_height = max_height
                new_width = int(width * ratio)
                image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        if 'scale' in resize_options:
            scale = resize_options['scale']
            new_width = int(width * scale)
            new_height = int(height * scale)
            image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        return image
    
    def _adjust_brightness(self, image: Image.Image, factor: float) -> Image.Image:
        """
        밝기 조정
        
        Args:
            image: 원본 이미지
            factor: 밝기 조정 계수 (0.5-2.0)
            
        Returns:
            PIL.Image: 밝기 조정된 이미지
        """
        enhancer = ImageEnhance.Brightness(image)
        return enhancer.enhance(factor)
    
    def _adjust_contrast(self, image: Image.Image, factor: float) -> Image.Image:
        """
        대비 조정
        
        Args:
            image: 원본 이미지
            factor: 대비 조정 계수 (0.5-2.0)
            
        Returns:
            PIL.Image: 대비 조정된 이미지
        """
        enhancer = ImageEnhance.Contrast(image)
        return enhancer.enhance(factor)
    
    def _adjust_sharpness(self, image: Image.Image, factor: float) -> Image.Image:
        """
        선명도 조정
        
        Args:
            image: 원본 이미지
            factor: 선명도 조정 계수 (0.5-2.0)
            
        Returns:
            PIL.Image: 선명도 조정된 이미지
        """
        enhancer = ImageEnhance.Sharpness(image)
        return enhancer.enhance(factor)
    
    def _denoise_image(self, image: Image.Image) -> Image.Image:
        """
        노이즈 제거
        
        Args:
            image: 원본 이미지
            
        Returns:
            PIL.Image: 노이즈 제거된 이미지
        """
        # PIL에서 간단한 노이즈 제거
        return image.filter(ImageFilter.MedianFilter(size=3))
    
    def _auto_adjust_brightness(self, image: Image.Image) -> Image.Image:
        """
        자동 밝기 조정
        
        Args:
            image: 원본 이미지
            
        Returns:
            PIL.Image: 자동 조정된 이미지
        """
        # 히스토그램 분석을 통한 자동 밝기 조정
        img_array = np.array(image)
        
        # 그레이스케일 변환
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array
        
        # 히스토그램 평활화
        equalized = cv2.equalizeHist(gray)
        
        # RGB로 변환
        if len(img_array.shape) == 3:
            result = cv2.cvtColor(equalized, cv2.COLOR_GRAY2RGB)
        else:
            result = equalized
        
        return Image.fromarray(result)
    
    def _encode_image(self, image: Image.Image, quality: str = 'medium') -> str:
        """
        이미지를 base64로 인코딩
        
        Args:
            image: 인코딩할 이미지
            quality: 품질 설정
            
        Returns:
            str: base64 인코딩된 이미지
        """
        # 임시 파일에 저장
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
            temp_filename = temp_file.name
        
        # 품질 설정 적용
        quality_settings = self.quality_settings.get(quality, self.quality_settings['medium'])
        
        # 이미지 저장
        image.save(temp_filename, 'JPEG', quality=quality_settings['quality'], optimize=True)
        
        # 파일 읽기 및 base64 인코딩
        with open(temp_filename, 'rb') as f:
            image_data = f.read()
        
        # 임시 파일 삭제
        os.unlink(temp_filename)
        
        return base64.b64encode(image_data).decode('utf-8')
    
    def create_thumbnail(self, image_data: bytes, size: Tuple[int, int] = (200, 200)) -> Dict[str, Any]:
        """
        썸네일 생성
        
        Args:
            image_data: 원본 이미지 데이터
            size: 썸네일 크기
            
        Returns:
            Dict: 썸네일 정보
        """
        try:
            image = self._load_image(image_data)
            if image is None:
                return {"success": False, "error": "이미지를 로드할 수 없습니다."}
            
            # 썸네일 생성
            image.thumbnail(size, Image.Resampling.LANCZOS)
            
            # 인코딩
            thumbnail_data = self._encode_image(image, 'low')
            
            return {
                "success": True,
                "thumbnail_data": thumbnail_data,
                "size": image.size
            }
            
        except Exception as e:
            logger.error(f"썸네일 생성 오류: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def extract_metadata(self, image_data: bytes) -> Dict[str, Any]:
        """
        이미지 메타데이터 추출
        
        Args:
            image_data: 이미지 데이터
            
        Returns:
            Dict: 메타데이터
        """
        try:
            image = Image.open(io.BytesIO(image_data))
            
            metadata = {
                "format": image.format,
                "mode": image.mode,
                "size": image.size,
                "width": image.width,
                "height": image.height,
                "file_size": len(image_data)
            }
            
            # EXIF 데이터 추출
            if hasattr(image, '_getexif') and image._getexif():
                exif = image._getexif()
                if exif:
                    metadata["exif"] = {
                        "orientation": exif.get(274, None),  # Orientation
                        "datetime": exif.get(36867, None),   # DateTime
                        "make": exif.get(271, None),         # Make
                        "model": exif.get(272, None)         # Model
                    }
            
            return metadata
            
        except Exception as e:
            logger.error(f"메타데이터 추출 오류: {str(e)}")
            return {"error": str(e)} 