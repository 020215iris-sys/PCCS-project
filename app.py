import os
import cv2
import mediapipe as mp


# ------------------------------
# FaceMesh 시각화를 위한 함수
# ------------------------------

mp_face_mesh = mp.solutions.face_mesh

def visualize_facemesh(image_path, save_path="face_mesh_output.jpg"):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"이미지를 찾을 수 없음: {image_path}")

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

        h, w, _ = img.shape
        for lm in result.multi_face_landmarks[0].landmark:
            cx = int(lm.x * w)
            cy = int(lm.y * h)
            cv2.circle(img, (cx, cy), 1, (0, 255, 0), -1)

        cv2.imwrite(save_path, img)
        print(f"FaceMesh 시각화 이미지 저장 완료 → {save_path}")
        return save_path

# 작업 디렉토리를 이미지가 있는 폴더로 변경
os.chdir("C:/PCCS/test_images")

# 이후에는 파일 이름만 사용 가능
visualize_facemesh("test.jpg", "face_mesh_output.jpg")

# ------------------------------
# 기존 로직 불러오기
# ------------------------------
from modules.palette_processor import load_all_palettes
from modules.face_detector import detect_face
from modules.skin_extractor import process_skin
from modules.season_classifier import classify_season


# ------------------------------
# 메인 기능
# ------------------------------
def main():
    print("팔레트 로딩 중...")
    palettes = load_all_palettes("C:/PCCS/palettes")

    img_path = input("이미지 경로를 입력하세요: ").strip()
    print("얼굴 인식 중...")
    _, face_crop, _ = detect_face(img_path)

    if face_crop is None:
        print("얼굴을 찾지 못했습니다.")
        return

    print("피부 색 추출 중...")
    skin_lab, _, _ = process_skin(img_path)

    if skin_lab is None:
        print("피부 색 추출 실패")
        return

    print("시즌 판정 중...")
    season = classify_season(skin_lab, palettes)
    print(f"판정된 시즌: {season}")
    print("face_box.png 파일을 확인해 주세요 (얼굴 위치 표시됨).")


# ------------------------------
# 실행부
# ------------------------------
if __name__ == "__main__":
    # 1) 퍼스널컬러 프로그램 실행
    main()

    # 2) FaceMesh 시각화 테스트 (원하면 활성화)
    visualize_facemesh("test.jpg", "face_mesh_output.jpg")
