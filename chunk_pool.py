"""
chunk_pool.py — select_recursive_*_chunk_span.xlsx(원본 청크 파일) 로더
"""

from __future__ import annotations

import ast
import os
import re

import pandas as pd

from query_pipeline import ChunkRef

# chunking_key -> 원본 파일 경로 (실제 배치 시 input/ 폴더 등으로 옮기고 경로만 바꾸면 됨)
DEFAULT_PATHS = {
    "recursive_300_100": "input/recursive_300_100_chunk_span.xlsx",
    "recursive_600_200": "input/recursive_600_200_chunk_span.xlsx",
    "recursive_1000_300": "input/recursive_1000_300_chunk_span.xlsx",
}

# source_doc의 사람이 읽을 문서명을 찾을 참조 파일들(rag_qa_2 밖 — 원본 문서 메타데이터
# 파이프라인 산출물). 순서대로 합치되 먼저 나온 파일의 이름이 우선한다. 두 파일 다
# uuid/doc_nm 컬럼을 갖는다. 없는 경로는 조용히 건너뛴다(참조 데이터가 없어도 파이프라인
# 자체는 돌아가야 하므로 — 그때는 doc_nm이 source_doc 해시로 남는다).
DOC_NAME_LOOKUP_PATHS = [
    "data/data_selected_v2.xlsx",
    "data/data_selected_v2.xlsx",
]

# source_doc은 크게 두 형태다: "{uuid16자리}_{chunk내부id}"(약관·상품설명서 등 대부분)와
# "감독규정_{hash}_{hash}"/"법령_{번호}_{날짜}_{hash}"(법령·규정류, 카테고리 접두어가 이미
# 붙어있어 그 자체로 어느 정도 읽을 수 있음). 문서명 조인은 앞의 uuid16자리 형태에만
# 적용한다 — 뒤의 두 형태는 참조 파일에 애초에 없는 별도 ID 체계다.
_UUID_SOURCE_DOC_RE = re.compile(r"^([0-9a-f]{16})_[0-9a-f]+$")


def _clean_content(raw) -> str:
    """빈 셀(NaN)을 빈 문자열로 바꾼다 - ChunkRef.content는 이후 전부(문자열 join, 정규식
    등)에서 항상 str이라고 가정하므로, 외부 파일을 읽는 이 경계에서 막아야 한다
    (실제로 recursive_300_100_chunk_span.xlsx에 content가 빈 행이 4건 있었다)."""
    return "" if pd.isna(raw) else str(raw)


def _parse_topic_list(raw) -> list[str]:
    if isinstance(raw, list):
        return raw
    try:
        return ast.literal_eval(raw)
    except (ValueError, SyntaxError, TypeError):
        return []


def _load_doc_name_lookup(paths: list[str] = None) -> dict:
    """uuid(source_doc 앞 16자리 hex) -> 사람이 읽을 문서명. DOC_NAME_LOOKUP_PATHS를
    순서대로 읽어 합치되, 먼저 나온 파일에 있는 이름이 우선한다(뒤 파일은 앞에서
    못 찾은 uuid만 채움)."""
    paths = paths if paths is not None else DOC_NAME_LOOKUP_PATHS
    lookup: dict = {}
    for path in paths:
        if not os.path.exists(path):
            continue
        df = pd.read_excel(path, usecols=["uuid", "doc_nm"])
        for uuid, doc_nm in zip(df["uuid"], df["doc_nm"]):
            if pd.notna(doc_nm) and uuid not in lookup:
                lookup[str(uuid)] = str(doc_nm)
    return lookup


def _resolve_doc_name(source_doc: str, name_lookup: dict) -> str:
    """source_doc이 uuid16자리 형태면 name_lookup에서 이름을 찾고, 없으면(또는 애초에
    다른 ID 체계면) source_doc을 그대로 doc_nm으로 쓴다."""
    m = _UUID_SOURCE_DOC_RE.match(source_doc)
    if not m:
        return source_doc
    return name_lookup.get(m.group(1), source_doc)


def load_chunk_pool(paths: dict = None, doc_name_lookup: dict = None) -> dict:
    """
    반환: pool[chunking_key] = list[ChunkRef]
    category는 원본 값 그대로 담는다 (6종 재태깅은 §3-3에서 별도 처리, 여기선 로딩만).
    경쟁사 문서는 원본 파일(input/*_chunk_span.xlsx)에서 이미 제거되어 있다 — 여기서는
    더 이상 걸러내지 않는다.
    doc_name_lookup을 안 주면(None) DOC_NAME_LOOKUP_PATHS에서 자동으로 읽는다 — 테스트
    등 외부 참조 파일에 의존하고 싶지 않은 호출부는 {}를 명시로 넘기면 된다.
    """
    paths = paths or DEFAULT_PATHS
    name_lookup = doc_name_lookup if doc_name_lookup is not None else _load_doc_name_lookup()
    pool = {}
    for chunking_key, path in paths.items():
        df = pd.read_excel(path)
        has_topic_list = "topic_list" in df.columns  # topic 태깅 전 원본 파일(_chunk.xlsx)엔 이 컬럼 자체가 없음

        chunks = [
            ChunkRef(
                chunk_id=row["chunk_id"],
                doc_id=row["source_doc"],
                doc_nm=_resolve_doc_name(row["source_doc"], name_lookup),
                category=row["category"],
                subcategory=row["subcategory"],
                content=_clean_content(row["content"]),
                topic_list=_parse_topic_list(row["topic_list"]) if has_topic_list else [],
                start_char=int(row["start_char"]),
                end_char=int(row["end_char"]),
            )
            for _, row in df.iterrows()
        ]
        pool[chunking_key] = chunks
    return pool


if __name__ == "__main__":
    pool = load_chunk_pool()
    for key, chunks in pool.items():
        print(f"{key}: {len(chunks)}개 청크, 예시 1건 -> {chunks[0].chunk_id} / {chunks[0].doc_nm}")
