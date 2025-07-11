from melo.api import TTS
import sounddevice as sd
import numpy as np


# 설정
speed = 1.0  # 음성 속도 조절
device = 'cpu'  # GPU 사용 시 'cuda' 또는 'cuda:0'
text = "안녕하세요."
model = TTS(language='KR', device=device)
speaker_ids = model.hps.data.spk2id

# 음성 생성 및 파일 저장
output_path = 'output.wav'
wav = model.tts_to_file(text, speaker_ids['KR'], output_path, speed=speed)
# 음성 재생
# with open(output_path, 'rb') as f:
#     wav = np.frombuffer(f.read(), dtype=np.int16)
# wav = wav / np.max(np.abs(wav))  # 정규화
# wav = wav.astype(np.float32)  # sounddevice는 float32 형식을 사용하
sd.play(wav, model.hps.data.sampling_rate)
sd.wait()

while True:
    text = input("Enter text to synthesize (or 'exit' to quit): ")
    if text.lower() == 'exit':
        break
    wav = model.tts_to_file(text, speaker_ids['KR'], output_path, speed=speed)
    sd.play(wav, model.hps.data.sampling_rate)
    sd.wait()