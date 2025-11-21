import cv2
import numpy as np
from PIL import Image
import mediapipe as mp

mp_face = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh

# Mediapipe 초기화
face_detection = mp_face.FaceDetection(model_selection=1, min_detection_confidence=0.5)
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5
)


# -------------------------------------------------
# 얼굴 탐지
# -------------------------------------------------
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

    # ==========================================
    # 🔥 추가된 부분: 얼굴 박스 시각화 + 저장
    # ==========================================
    img_box = img.copy()
    cv2.rectangle(img_box, (x1, y1), (x2, y2), (0, 255, 255), 2)

    save_path = "face_box.png"
    cv2.imwrite(save_path, img_box)
    print(f"[저장됨] 얼굴 박스 이미지 → {save_path}")
    # ==========================================

    # 얼굴 crop
    face_crop = img[y1:y2, x1:x2]

    return img, face_crop, (x1, y1, x2, y2)


# -------------------------------------------------
# 입술 합성 함수 (원본 mesh → crop 변환)
# -------------------------------------------------
def apply_lip_color(img, face_crop, bbox, lip_color=(255, 0, 0), alpha=0.6):
    x1, y1, x2, y2 = bbox

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(img_rgb)

    if not results.multi_face_landmarks:
        return face_crop

    landmarks = results.multi_face_landmarks[0].landmark

    # Mediapipe 입술 랜드마크
    lip_idx = list(range(61, 88)) + list(range(291, 318))

    # crop 기준으로 좌표 변환
    lip_points_crop = []
    h, w, _ = img.shape
    for idx in lip_idx:
        lx = int(landmarks[idx].x * w)
        ly = int(landmarks[idx].y * h)
        lip_points_crop.append((lx - x1, ly - y1))

    lip_points_crop = np.array(lip_points_crop)

    # 마스크 생성
    mask = np.zeros_like(face_crop, dtype=np.uint8)
    cv2.fillPoly(mask, [lip_points_crop.astype(np.int32)], (255, 255, 255))

    # 립 색상 (BGR)
    color_layer = np.zeros_like(face_crop)
    color_layer[:] = lip_color[::-1]

    # 색 적용
    lip_colored = cv2.addWeighted(face_crop, 1-alpha, color_layer, alpha, 0)

    # 마스크로 합성
    output = face_crop.copy()
    output[mask[:, :, 0] > 0] = lip_colored[mask[:, :, 0] > 0]

    return output


# -------------------------------------------------
# before/after 출력
# -------------------------------------------------
def generate_before_after(image_path, lip_color=(255, 0, 128)):
    original_img, face_crop, bbox = detect_face(image_path)
    if face_crop is None:
        print("얼굴이 감지되지 않았습니다.")
        return

    lip_applied = apply_lip_color(original_img, face_crop, bbox, lip_color)

    # face_crop 영역에 덮어쓰기
    x1, y1, x2, y2 = bbox
    after_img = original_img.copy()
    after_img[y1:y2, x1:x2] = lip_applied

    before = Image.fromarray(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB))
    after = Image.fromarray(cv2.cvtColor(after_img, cv2.COLOR_BGR2RGB))

    combined = Image.new('RGB', (before.width*2, before.height))
    combined.paste(before, (0,0))
    combined.paste(after, (before.width, 0))

    combined.save("before_after_fixed.jpg")
    print("저장 완료: before_after_fixed.jpg")


if __name__ == "__main__":
    generate_before_after("test.jpg", lip_color=(255, 0, 128))

import cv2
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh

def visualize_facemesh(image_path, save_path="face_mesh_result.jpg"):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError("이미지를 찾을 수 없음")

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5
    ) as mesh:

        result = mesh.process(img_rgb)

        if not result.multi_face_landmarks:
            print("FaceMesh 감지 실패")
            return None

        landmarks = result.multi_face_landmarks[0]
        h, w, _ = img.shape

        # 얼굴 랜드마크 그리기
        for lm in landmarks.landmark:
            x = int(lm.x * w)
            y = int(lm.y * h)
            cv2.circle(img, (x, y), 1, (0, 255, 0), -1)

        cv2.imwrite(save_path, img)
        print(f"FaceMesh 시각화 이미지 저장됨 → {save_path}")

        return save_path
