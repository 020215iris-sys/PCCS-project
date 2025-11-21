import os
import csv
import requests
from datetime import datetime
from bs4 import BeautifulSoup


# ---------------------------------------------------
# 1. 이미지 저장 함수
# ---------------------------------------------------
def download_image(url, save_path):
    try:
        img = requests.get(url, timeout=10)
        if img.status_code == 200:
            with open(save_path, "wb") as f:
                f.write(img.content)
            return True
    except:
        return False
    return False


# ---------------------------------------------------
# 2. HTML 파일에서 정보 추출
# ---------------------------------------------------
def parse_lip_html(html_path):
    with open(html_path, "r", encoding="utf-8") as f:
        html = f.read()

    soup = BeautifulSoup(html, "html.parser")

    # 브랜드명
    brand = "UnknownBrand"
    brand_tag = soup.select_one(".tx_brand, .prd_brand")
    if brand_tag:
        brand = brand_tag.get_text(strip=True)

    # 제품명
    product_name = "UnknownProduct"
    name_tag = soup.select_one(".prd_name, .product_tit")
    if name_tag:
        product_name = name_tag.get_text(strip=True)

    # 컬러칩 이미지들.
    chip_imgs = soup.select(".ColorChips_colorchip-item__PXPll img")
    image_urls = [img["src"] for img in chip_imgs if img.get("src")]

    return brand, product_name, image_urls


# ---------------------------------------------------
# 3. 메인 실행 (HTML → 이미지 저장 + CSV 기록)
# ---------------------------------------------------
def run_from_html(html_path):
    BASE = os.path.dirname(os.path.dirname(__file__))  # /PCCS
    RESULT_DIR = os.path.join(BASE, "result")
    os.makedirs(RESULT_DIR, exist_ok=True)

    # 이미지 저장 폴더
    COLORCHIP_DIR = os.path.join(RESULT_DIR, "colorchips")
    os.makedirs(COLORCHIP_DIR, exist_ok=True)

    CSV_PATH = os.path.join(RESULT_DIR, "lip_info.csv")

    # HTML 분석
    brand, product_name, image_urls = parse_lip_html(html_path)

    # 고유번호 = html 파일명에서 확장자 제거
    product_id = os.path.splitext(os.path.basename(html_path))[0]

    rows = []

    # 이미지 다운로드 및 파일명 생성
    for idx, url in enumerate(image_urls, start=1):
        filename = f"{brand}_{product_name}_{product_id}_chip{idx}.jpg"
        filename = filename.replace("/", "_").replace(" ", "_")
        save_path = os.path.join(COLORCHIP_DIR, filename)

        download_image(url, save_path)

        rows.append({
            "brand": brand,
            "product_name": product_name,
            "product_id": product_id,
            "image_url": url,
            "saved_filename": filename,
            "crawled_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })

    # CSV 저장 (중복 헤더 방지)
    header = ["brand", "product_name", "product_id", "image_url", "saved_filename", "crawled_at"]
    write_header = not os.path.exists(CSV_PATH)

    with open(CSV_PATH, "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)

    print(f"\n✔ 브랜드: {brand}")
    print(f"✔ 제품명: {product_name}")
    print(f"✔ 저장된 컬러칩 이미지 수: {len(image_urls)}")
    print(f"✔ 이미지 저장 위치: {COLORCHIP_DIR}")
    print(f"✔ CSV 저장 위치: {CSV_PATH}")
    print("🎉 완료!")


# ---------------------------------------------------
# 4. 직접 실행할 때
# ---------------------------------------------------
if __name__ == "__main__":
    # 예: 나연이 저장한 HTML 파일
    html_file = "lip_page.html"  # 파일명을 여기에 입력
    run_from_html(html_file)
