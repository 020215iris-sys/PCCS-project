import os
import csv
import requests
import re
from datetime import datetime
from bs4 import BeautifulSoup

# ============================================
# 1) 경로 설정
# ============================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULT_DIR = os.path.join(BASE_DIR, "result")
IMG_DIR = os.path.join(RESULT_DIR, "colorchips")
CSV_PATH = os.path.join(RESULT_DIR, "lip_info.csv")

os.makedirs(IMG_DIR, exist_ok=True)

# ============================================
# 파일명 금지문자 및 길이 제한 처리 함수
# ============================================
def clean_filename(text):
    # 금지문자 제거
    text = re.sub(r'[\\/:*?"<>|]', '', text)
    # 줄바꿈 제거
    text = text.replace("\n", "").replace("\r", "")
    # 공백 정리
    text = text.strip()
    # 파일명 길이 제한 (윈도우 260자 오류 방지)
    return text[:90]

# ============================================
# 2) HTML 파일 파싱 함수
# ============================================
def parse_from_html(html_path):

    with open(html_path, "r", encoding="utf-8") as f:
        html = f.read()

    soup = BeautifulSoup(html, "html.parser")

    # 브랜드명 찾기
    brand_tag = soup.select_one(".TopUtils_btn-brand__tvEdp, .prd_brand, .tx_brand")
    brand = brand_tag.text.strip() if brand_tag else "UnknownBrand"

    # 제품명 찾기
    name_tag = soup.select_one(".prd_name, .product_tit, h3")
    product_name = name_tag.text.strip() if name_tag else "UnknownProduct"

    # 컬러칩 이미지 URL 수집
    chips = soup.select(".ColorChips_colorchip-item__PXPll img")
    color_list = []

    for img in chips:
        alt_name = img.get("alt", "UnknownColor").strip()
        img_url = img.get("src", "")

        if img_url:
            color_list.append((alt_name, img_url))

    return brand, product_name, color_list


# ============================================
# 3) 이미지 저장 함수 (안전 파일명 적용)
# ============================================
def save_image(img_url, brand, color_name):
    try:
        safe_brand = clean_filename(brand)
        safe_color = clean_filename(color_name)

        file_name = f"{safe_brand}_{safe_color}.jpg"
        save_path = os.path.join(IMG_DIR, file_name)

        response = requests.get(img_url, timeout=10)
        response.raise_for_status()

        with open(save_path, "wb") as f:
            f.write(response.content)

        return file_name

    except Exception as e:
        print(f"이미지 저장 실패: {e}")
        return None


# ============================================
# 4) CSV 생성/추가 함수
# ============================================
def save_to_csv(rows):

    header = ["brand", "product_name", "color_name", "img_url", "img_file", "date", "time"]

    write_header = not os.path.exists(CSV_PATH)

    with open(CSV_PATH, "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)

        if write_header:
            writer.writerow(header)

        writer.writerows(rows)


# ============================================
# 5) 실행 메인 함수
# ============================================
def run(html_path):
    print("크롤링 시작…")

    brand, product_name, color_list = parse_from_html(html_path)

    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M:%S")

    rows = []

    for color_name, img_url in color_list:
        img_file = save_image(img_url, brand, color_name)

        rows.append([
            brand,
            product_name,
            color_name,
            img_url,
            img_file,
            date_str,
            time_str
        ])

    save_to_csv(rows)

    print(f"CSV 저장 완료 → {CSV_PATH}")
    print("이미지 저장 완료 → colorchips 폴더")
    print("크롤링 완료! 💜")


# ============================================
# 6) 직접 실행할 때
# ============================================
if __name__ == "__main__":
    test_html = r"C:\Users\user\Desktop\learning4\PCCS-project\products\html\lipfull12mode.html"
    run(test_html)
