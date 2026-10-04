from pathlib import Path

import cv2

from services.ocr_service import read_text

def extract_full_name(image_path: str) -> str | None:
    image = cv2.imread(image_path)

    if image is None:
        raise ValueError(f"Cannot read image: {image_path}")

    height, width = image.shape[:2]

    # Vùng Họ và tên, tính theo tỷ lệ của ảnh
    x1 = int(width * 0.27)
    y1 = int(height * 0.53)
    x2 = int(width * 0.61)
    y2 = int(height * 0.63)

    crop = image[y1:y2, x1:x2]

    if crop.size == 0:
        raise ValueError("Invalid full-name crop region")

    crop_path = Path(image_path).with_name("full_name_crop.png")

    success = cv2.imwrite(str(crop_path), crop)

    if not success:
        raise ValueError("Cannot save full-name crop")

    ocr_results = read_text(str(crop_path))

    candidates = [
        item["text"].strip()
        for item in ocr_results
        if item["confidence"] >= 0.5
    ]

    if not candidates:
        return None

    return " ".join(candidates)
def extract_date_of_birth(image_path: str) -> str | None:
    image = cv2.imread(image_path)

    if image is None:
        raise ValueError(f"Cannot read image: {image_path}")

    height, width = image.shape[:2]

    # Vùng Ngày sinh
    x1 = int(width * 0.56)
    y1 = int(height * 0.61)
    x2 = int(width * 0.76)
    y2 = int(height * 0.70)

    crop = image[y1:y2, x1:x2]

    if crop.size == 0:
        raise ValueError("Invalid date-of-birth crop region")

    crop_path = Path(image_path).with_name("date_of_birth_crop.png")

    success = cv2.imwrite(str(crop_path), crop)

    if not success:
        raise ValueError("Cannot save date-of-birth crop")

    ocr_results = read_text(str(crop_path))

    for item in ocr_results:
        text = item["text"].strip()

        if item["confidence"] >= 0.5:
            return text

    return None
def extract_place_of_origin(image_path: str) -> str | None:
    image = cv2.imread(image_path)

    if image is None:
        raise ValueError(f"Cannot read image: {image_path}")

    height, width = image.shape[:2]

    # Vùng Quê quán
    x1 = int(width * 0.09)
    y1 = int(height * 0.78)
    x2 = int(width * 0.94)
    y2 = int(height * 0.88)

    crop = image[y1:y2, x1:x2]

    if crop.size == 0:
        raise ValueError("Invalid place-of-origin crop region")

    crop_path = Path(image_path).with_name("place_of_origin_crop.png")

    success = cv2.imwrite(str(crop_path), crop)

    if not success:
        raise ValueError("Cannot save place-of-origin crop")

    ocr_results = read_text(str(crop_path))

    candidates = [
        item["text"].strip()
        for item in ocr_results
        if item["confidence"] >= 0.5
    ]

    if not candidates:
        return None

    return ", ".join(candidates)
def extract_residence(image_path: str) -> str | None:
    image = cv2.imread(image_path)

    if image is None:
        raise ValueError(f"Cannot read image: {image_path}")

    # Ảnh test hiện tại có kích thước khoảng 1061 x 686.
    # Hai vùng này đã được kiểm tra trực tiếp bằng OCR.

    # Dòng 1: Thôn Bích Đông
    first_line = image[590:630, 280:1000]

    # Dòng 2 + 3:
    # Nam Chính
    # Nam Sách, Hải Dương
    remaining_lines = image[620:680, 280:1000]

    first_path = Path(image_path).with_name(
        "residence_first_line.png"
    )

    remaining_path = Path(image_path).with_name(
        "residence_remaining_lines.png"
    )

    if not cv2.imwrite(str(first_path), first_line):
        raise ValueError("Cannot save residence first line")

    if not cv2.imwrite(str(remaining_path), remaining_lines):
        raise ValueError("Cannot save residence remaining lines")

    first_results = read_text(str(first_path))
    remaining_results = read_text(str(remaining_path))

    first_candidates = [
        item["text"].strip()
        for item in first_results
        if item["confidence"] >= 0.5
    ]

    remaining_candidates = [
        item["text"].strip()
        for item in remaining_results
        if item["confidence"] >= 0.5
    ]

    lines = first_candidates + remaining_candidates

    if not lines:
        return None
    if len(lines) >= 2 and lines[0].lower().startswith("thôn"):
        lines[0] = f"{lines[0]} {lines[1]}"
        lines.pop(1)
    return ", ".join(lines)
def extract_nationality(image_path: str) -> str | None:
    image = cv2.imread(image_path)

    if image is None:
        raise ValueError(f"Cannot read image: {image_path}")

    # Vùng Quốc tịch
    image = image[
        int(image.shape[0] * 0.66):int(image.shape[0] * 0.76),
        int(image.shape[1] * 0.78):int(image.shape[1] * 0.95),
    ]

    crop_path = Path(image_path).with_name(
        "nationality_crop.png"
    )

    if not cv2.imwrite(str(crop_path), image):
        raise ValueError("Cannot save nationality crop")

    ocr_results = read_text(str(crop_path))

    for item in ocr_results:
        if item["confidence"] >= 0.5:
            text = item["text"].strip()

            if "việt" in text.lower() or "viet" in text.lower():
                return "Việt Nam"

    return None
def extract_gender(image_path: str) -> str | None:
    image = cv2.imread(image_path)

    if image is None:
        raise ValueError(f"Cannot read image: {image_path}")

    height, width = image.shape[:2]

    crop = image[
        int(height * 0.60):int(height * 0.82),
        int(width * 0.45):int(width * 0.90),
    ]

    crop_path = Path(image_path).with_name(
        "gender_crop.png"
    )

    if not cv2.imwrite(str(crop_path), crop):
        raise ValueError("Cannot save gender crop")

    ocr_results = read_text(str(crop_path))

    for item in ocr_results:
        text = item["text"].strip().lower()
        confidence = item["confidence"]

        if confidence >= 0.5:
            if text == "nam":
                return "Nam"

            if text in {"nữ", "nu"}:
                return "Nữ"

    return None
def extract_cccd_fields(image_path: str) -> dict:
    return {
        "full_name": extract_full_name(image_path),
        "date_of_birth": extract_date_of_birth(image_path),
        "gender": extract_gender(image_path),
        "nationality": extract_nationality(image_path),
        "place_of_origin": extract_place_of_origin(image_path),
        "residence": extract_residence(image_path),
    }
