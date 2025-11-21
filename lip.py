import cv2
import numpy as np
from PIL import Image
import mediapipe as mp

# ------------------------------
# Mediapipe 초기화
# ------------------------------
mp_face = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh

face_detection = mp_face.FaceDetection(model_selection=1, min_detection_confidence=0.5)
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1,
                                  refine_landmarks=True, min_detection_confidence=0.5)

# ------------------------------
# 얼굴 검출 함수
# ------------------------------
def detect_face(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"이미지를 찾을 수 없음: {image_path}")

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = face_detection.process(img_rgb)

    if not result.detections:
        return img, None, None

    det = result.detections[0]
    bbox = det.location_data.relative_bounding_box
    h, w, _ = img.shape
    x1 = int(bbox.xmin * w)
    y1 = int(bbox.ymin * h)
    x2 = int((bbox.xmin + bbox.width) * w)
    y2 = int((bbox.ymin + bbox.height) * h)
    face_crop = img[y1:y2, x1:x2]

    return img, face_crop, (x1, y1, x2, y2)

# ------------------------------
# 입술 색상 합성 함수
# ------------------------------
def apply_lip_color(face_img, lip_color=(255, 0, 0), alpha=0.6):
    img_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(img_rgb)

    if not results.multi_face_landmarks:
        return face_img

    landmarks = results.multi_face_landmarks[0].landmark
    lips_idx = list(range(61, 88)) + list(range(291, 318))  # 입술 점 인덱스
    h, w, _ = face_img.shape
    lip_points = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in lips_idx]

    # 입술 마스크 생성
    mask = np.zeros_like(face_img, dtype=np.uint8)
    cv2.fillPoly(mask, [np.array(lip_points, dtype=np.int32)], (255, 255, 255))

    # 립 색상 적용
    color_layer = np.zeros_like(face_img, dtype=np.uint8)
    color_layer[:] = lip_color[::-1]  # OpenCV는 BGR
    lip_colored = cv2.bitwise_and(color_layer, mask)

    # 합성
    inv_mask = cv2.bitwise_not(mask)
    background = cv2.bitwise_and(face_img, inv_mask)
    combined = cv2.addWeighted(background + lip_colored, 1.0, face_img, 0, 0)

    return combined

# ------------------------------
# 비포/애프터 합성 및 저장
# ------------------------------
def generate_before_after(image_path, lip_color=(255, 0, 128)):
    original_img, face_crop, bbox = detect_face(image_path)
    if face_crop is None:
        print("얼굴이 감지되지 않았습니다.")
        return

    # 립 합성
    lip_applied = apply_lip_color(face_crop, lip_color=lip_color)

    # 원본 이미지에 다시 붙이기
    x1, y1, x2, y2 = bbox
    after_img = original_img.copy()
    after_img[y1:y2, x1:x2] = lip_applied

    # PIL로 나란히 비교
    before = Image.fromarray(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB))
    after = Image.fromarray(cv2.cvtColor(after_img, cv2.COLOR_BGR2RGB))
    combined = Image.new('RGB', (before.width*2, before.height))
    combined.paste(before, (0, 0))
    combined.paste(after, (before.width, 0))

    combined.show()
    combined.save("before_after_result.jpg")
    print("비포/애프터 이미지 저장 완료: before_after_result.jpg")

# ------------------------------
# 실행
# ------------------------------
if __name__ == "__main__":
    generate_before_after("test.jpg", lip_color=(255, 0, 128))  # 원하는 립 색상 RGB
