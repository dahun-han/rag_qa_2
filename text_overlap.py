"""text_overlap.py — 질문/청크 간 키워드 겹침 계산 (임베딩 API 미사용).

retrieval_sim.py(모의 검색)와 candidate_pool.py(Hard/Easy Neg 등급 판정)가 공유해서
쓰는 저비용 겹침 점수 함수. 형태소 분석기는 쓰지 않는 러프한 1차 구현.
"""

from __future__ import annotations

import re

_TOKEN_RE = re.compile(r"[가-힣]{2,}|[A-Za-z0-9]{2,}")

# 조사·어미 등 의미 없는 토큰이 우연히 2글자 이상으로 잡히는 경우를 걸러낸다.
_STOPWORDS = {
    "이것", "그것", "저것", "합니다", "습니다", "하는", "있는", "없는", "에서", "으로",
    "그리고", "하지만", "때문", "경우", "대해", "위해", "통해",
}


def tokenize(text: str) -> set[str]:
    return {t for t in _TOKEN_RE.findall(text or "") if t not in _STOPWORDS}


def overlap_score(question_tokens: set[str], chunk_content: str) -> float:
    """질문 토큰이 청크 content에 얼마나 커버되는지 비율. 질문 토큰이 하나도
    없으면(빈 질문 등) 0.0을 반환한다."""
    if not question_tokens:
        return 0.0
    chunk_tokens = tokenize(chunk_content)
    return len(question_tokens & chunk_tokens) / len(question_tokens)
