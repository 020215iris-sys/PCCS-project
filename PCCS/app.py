import os
from modules.palette_processor import load_all_palettes
from modules.face_detector import detect_face
from modules.skin_extractor import process_skin
from modules.season_classifier import classify_season

def main():
    print("팔레트 로딩 중...")
    palettes = load_all_palettes("C:/PCCS/palettes")

    img_path = input("이미지 경로를 입력하세요: ").strip()
    print("얼굴 인식 중...")
    _, face_crop = detect_face(img_path)

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

if __name__ == "__main__":
    main()
