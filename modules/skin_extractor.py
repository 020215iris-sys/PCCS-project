import cv2
import numpy as np
from skimage import color

def white_balance(img, strength=0.2):
    """
    단순 화이트 밸런스(LAB 색공간 기준)
    strength: 0~1, 보정 강도
    """
    # LAB 변환 후 float32로
    result = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    avg_a = np.mean(result[:, :, 1])
    avg_b = np.mean(result[:, :, 2])
    
    # 채널 보정
    result[:, :, 1] -= (avg_a - 128) * strength
    result[:, :, 2] -= (avg_b - 128) * strength
    
    # 0~255 범위 제한 후 uint8로
    result = np.clip(result, 0, 255).astype(np.uint8)
    
    # 다시 BGR로 변환
    return cv2.cvtColor(result, cv2.COLOR_LAB2BGR)


def extract_skin_mask(img):
    """
    피부 영역 마스크 추출
    """
    # YCrCb 색공간에서 피부 영역 추출
    img_ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    lower = np.array([0, 135, 85], dtype=np.uint8)
    upper = np.array([255, 180, 135], dtype=np.uint8)
    mask = cv2.inRange(img_ycrcb, lower, upper)
    return mask


def process_skin(image_path):
    """
    이미지 경로 입력 → 피부 색상 추출(LAB) + 화이트밸런스
    반환: skin_lab (np.array), corrected_img, skin_mask
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"이미지를 찾을 수 없습니다: {image_path}")
    
    # 화이트 밸런스 적용
    corrected = white_balance(img, 0.2)
    
    # 피부 마스크
    skin_mask = extract_skin_mask(corrected)
    
    # 피부 영역 평균 색상 계산
    skin_pixels = corrected[skin_mask > 0]
    if len(skin_pixels) == 0:
        skin_lab = None
    else:
        skin_lab = color.rgb2lab(skin_pixels.reshape(-1, 1, 3)).reshape(-1, 3).mean(axis=0)
    
    return skin_lab, corrected, skin_mask
