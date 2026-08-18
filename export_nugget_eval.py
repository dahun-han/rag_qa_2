"""export_nugget_eval.py — output_*.xlsx(QA 시트)를 nugget 1행=1건짜리
평가용 long-format으로 변환한다. Item ID~Vital까지만 생성한다
(그 뒤 gpt-4.1-mini~사람 검수 메모 등은 평가팀이 채점 시 채우는 컬럼이라 여기선 안 만듦).

사용법:
    python export_nugget_eval.py output/output_recursive_600_200.xlsx
    -> output/output_recursive_600_200_nugget_eval.xlsx 생성
"""

from __future__ import annotations

import json
import sys

import pandas as pd

EVAL_COLUMNS = [
    "Item ID", "Data Type", "Nugget ID", "Nugget Key", "Nugget Value",
    "Ground Truth", "Query", "Nugget Type", "Vital",
]


def flatten_nuggets_for_eval(qa_xlsx_path: str) -> pd.DataFrame:
    qa_df = pd.read_excel(qa_xlsx_path, sheet_name="QA")
    rows = []
    for _, row in qa_df.iterrows():
        nuggets = json.loads(row["nuggets"])
        for i, n in enumerate(nuggets):
            rows.append({
                "Item ID": row["index"],
                "Data Type": row["data_type"],
                "Nugget ID": n["id"],
                "Nugget Key": n["key"],
                "Nugget Value": n["value"],
                # 참고 파일과 동일하게, Ground Truth/Query는 그 문항의 첫 nugget 행에만 채운다.
                "Ground Truth": row["ground_truth"] if i == 0 else None,
                "Query": row["query"] if i == 0 else None,
                "Nugget Type": n["type"],
                "Vital": "Y" if n.get("vital", True) else "N",
            })
    return pd.DataFrame(rows, columns=EVAL_COLUMNS)


def main():
    if len(sys.argv) != 2:
        print("사용법: python export_nugget_eval.py <output_*.xlsx 경로>")
        sys.exit(1)

    src_path = sys.argv[1]
    out_path = src_path.rsplit(".xlsx", 1)[0] + "_nugget_eval.xlsx"

    df = flatten_nuggets_for_eval(src_path)
    df.to_excel(out_path, index=False)
    print(f"저장 완료: {out_path} ({len(df)}행, 문항 {df['Item ID'].nunique()}건)")


if __name__ == "__main__":
    main()
