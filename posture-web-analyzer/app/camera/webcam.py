import cv2

def open_camera(preferred_indexes=[0, 1, 2], width=1280, height=720):
    """
    여러 인덱스를 시도하여 웹캠을 열고, 해상도를 설정합니다.
    성공 시 cv2.VideoCapture 객체를 반환, 실패 시 None 반환.
    """
    cap = None
    for camera_index in preferred_indexes:
        cap = cv2.VideoCapture(camera_index)
        if cap.isOpened():
            print(f"웹캠 {camera_index}에 연결되었습니다.")
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            return cap
        else:
            cap.release()
    print("웹캠에 연결할 수 없습니다. 테스트용 비디오 파일을 사용하거나 웹캠 권한을 확인해주세요.")
    return None

def read_frame(cap):
    """
    웹캠에서 프레임을 읽어옵니다.
    성공 시 (ret, frame) 반환, 실패 시 (False, None) 반환.
    """
    if cap is None or not cap.isOpened():
        return False, None
    return cap.read()

def release_camera(cap):
    """
    웹캠 자원을 해제합니다.
    """
    if cap is not None and cap.isOpened():
        cap.release()