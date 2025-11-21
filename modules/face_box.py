import cv2
import mediapipe as mp

mp_face = mp.solutions.face_detection

def save_face_box(image_path, save_path="face_box.png"):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"이미지를 찾을 수 없습니다: {image_path}")

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    with mp_face.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face:
        result = face.process(img_rgb)

    if not result.detections:
        print("얼굴 감지 실패")
        return None

    det = result.detections[0]
    bbox = det.location_data.relative_bounding_box

    h, w, _ = img.shape
    x1 = int(bbox.xmin * w)
    y1 = int(bbox.ymin * h)
    x2 = int((bbox.xmin + bbox.width) * w)
    y2 = int((bbox.ymin + bbox.height) * h)

    # 얼굴 박스 그리기
    boxed = img.copy()
    cv2.rectangle(boxed, (x1, y1), (x2, y2), (0, 255, 0), 3)

    cv2.imwrite(save_path, boxed)
    return save_path
