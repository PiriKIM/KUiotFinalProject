import pyttsx3
import gtts
from gtts import gTTS
import tempfile
import os
import base64
import logging
from typing import Optional, Dict, Any
import asyncio
from threading import Thread

logger = logging.getLogger(__name__)

class TTSService:
    """TTS (Text-to-Speech) 서비스"""
    
    def __init__(self):
        """TTS 엔진 초기화"""
        self.engine = None
        self.language = 'ko'
        self.voice_rate = 150
        self.voice_volume = 0.8
        
        try:
            # pyttsx3 엔진 초기화 (오프라인)
            self.engine = pyttsx3.init()
            self.engine.setProperty('rate', self.voice_rate)
            self.engine.setProperty('volume', self.voice_volume)
            
            # 한국어 음성 설정
            voices = self.engine.getProperty('voices')
            for voice in voices:
                if 'korean' in voice.name.lower() or 'ko' in voice.id.lower():
                    self.engine.setProperty('voice', voice.id)
                    break
            
            logger.info("TTS 서비스 초기화 완료 (pyttsx3)")
            
        except Exception as e:
            logger.warning(f"pyttsx3 초기화 실패: {str(e)}")
            self.engine = None
    
    async def speak_text(self, text: str, use_online: bool = False) -> Dict[str, Any]:
        """
        텍스트를 음성으로 변환
        
        Args:
            text: 변환할 텍스트
            use_online: 온라인 TTS 사용 여부 (gTTS)
            
        Returns:
            Dict: TTS 결과
        """
        try:
            if use_online or not self.engine:
                return await self._speak_online(text)
            else:
                return await self._speak_offline(text)
                
        except Exception as e:
            logger.error(f"TTS 변환 중 오류: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "audio_data": None
            }
    
    async def _speak_offline(self, text: str) -> Dict[str, Any]:
        """
        오프라인 TTS (pyttsx3)
        
        Args:
            text: 변환할 텍스트
            
        Returns:
            Dict: TTS 결과
        """
        try:
            # 임시 파일 생성
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                temp_filename = temp_file.name
            
            # 음성 생성
            self.engine.save_to_file(text, temp_filename)
            self.engine.runAndWait()
            
            # 오디오 파일 읽기
            with open(temp_filename, 'rb') as audio_file:
                audio_data = audio_file.read()
            
            # base64 인코딩
            audio_base64 = base64.b64encode(audio_data).decode('utf-8')
            
            # 임시 파일 삭제
            os.unlink(temp_filename)
            
            return {
                "success": True,
                "audio_data": audio_base64,
                "format": "wav",
                "method": "offline"
            }
            
        except Exception as e:
            logger.error(f"오프라인 TTS 오류: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "audio_data": None
            }
    
    async def _speak_online(self, text: str) -> Dict[str, Any]:
        """
        온라인 TTS (gTTS)
        
        Args:
            text: 변환할 텍스트
            
        Returns:
            Dict: TTS 결과
        """
        try:
            # gTTS 객체 생성
            tts = gTTS(text=text, lang=self.language, slow=False)
            
            # 임시 파일 생성
            with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as temp_file:
                temp_filename = temp_file.name
            
            # 음성 생성
            tts.save(temp_filename)
            
            # 오디오 파일 읽기
            with open(temp_filename, 'rb') as audio_file:
                audio_data = audio_file.read()
            
            # base64 인코딩
            audio_base64 = base64.b64encode(audio_data).decode('utf-8')
            
            # 임시 파일 삭제
            os.unlink(temp_filename)
            
            return {
                "success": True,
                "audio_data": audio_base64,
                "format": "mp3",
                "method": "online"
            }
            
        except Exception as e:
            logger.error(f"온라인 TTS 오류: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "audio_data": None
            }
    
    def speak_realtime(self, text: str) -> bool:
        """
        실시간 음성 출력 (블로킹)
        
        Args:
            text: 출력할 텍스트
            
        Returns:
            bool: 성공 여부
        """
        try:
            if self.engine:
                # 별도 스레드에서 실행
                thread = Thread(target=self._speak_thread, args=(text,))
                thread.daemon = True
                thread.start()
                return True
            else:
                logger.warning("TTS 엔진이 초기화되지 않았습니다.")
                return False
                
        except Exception as e:
            logger.error(f"실시간 TTS 오류: {str(e)}")
            return False
    
    def _speak_thread(self, text: str):
        """TTS 스레드 함수"""
        try:
            self.engine.say(text)
            self.engine.runAndWait()
        except Exception as e:
            logger.error(f"TTS 스레드 오류: {str(e)}")
    
    def generate_posture_feedback_audio(self, posture_score: float, feedback: str) -> Dict[str, Any]:
        """
        자세 피드백 음성 생성
        
        Args:
            posture_score: 자세 점수
            feedback: 피드백 메시지
            
        Returns:
            Dict: 음성 결과
        """
        try:
            # 점수에 따른 음성 메시지 생성
            if posture_score >= 90:
                message = f"완벽한 자세입니다! 점수는 {posture_score}점입니다."
            elif posture_score >= 80:
                message = f"좋은 자세입니다. 점수는 {posture_score}점입니다. {feedback}"
            elif posture_score >= 70:
                message = f"자세를 개선해보세요. 점수는 {posture_score}점입니다. {feedback}"
            else:
                message = f"자세를 바로잡아주세요. 점수는 {posture_score}점입니다. {feedback}"
            
            # 비동기로 음성 생성
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(self.speak_text(message))
            loop.close()
            
            return result
            
        except Exception as e:
            logger.error(f"피드백 음성 생성 오류: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "audio_data": None
            }
    
    def set_voice_properties(self, rate: Optional[int] = None, 
                           volume: Optional[float] = None,
                           language: Optional[str] = None):
        """
        음성 속성 설정
        
        Args:
            rate: 음성 속도 (기본값: 150)
            volume: 음성 볼륨 (0.0-1.0, 기본값: 0.8)
            language: 언어 코드 (기본값: 'ko')
        """
        if rate is not None:
            self.voice_rate = rate
            if self.engine:
                self.engine.setProperty('rate', rate)
        
        if volume is not None:
            self.voice_volume = max(0.0, min(1.0, volume))
            if self.engine:
                self.engine.setProperty('volume', self.voice_volume)
        
        if language is not None:
            self.language = language
    
    def get_available_voices(self) -> list:
        """
        사용 가능한 음성 목록 반환
        
        Returns:
            list: 음성 목록
        """
        if not self.engine:
            return []
        
        voices = self.engine.getProperty('voices')
        return [
            {
                "id": voice.id,
                "name": voice.name,
                "languages": voice.languages,
                "gender": voice.gender
            }
            for voice in voices
        ]
    
    def __del__(self):
        """리소스 정리"""
        if self.engine:
            self.engine.stop() 