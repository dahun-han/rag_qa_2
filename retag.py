"""retag.py — 원본 category 값을 사용자 확정 6종 코드로 재태깅"""

CATEGORY_CODE = {"여신": "LS", "수신": "DS", "외환": "FX", "내부통제": "CD", "자산관리": "WM", "고객관리": "CR"}
CODE_TO_CATEGORY = {v: k for k, v in CATEGORY_CODE.items()}

# 원본 파일에 등장하는 category 원문 -> 6종 name
RAW_TO_NAME = {
    "여신": "여신",
    "수신": "수신",
    "외환": "외환",
    "자산관리": "자산관리",
    "고객중심": "고객관리",
    "법률 및 내부통제": "내부통제",
}
