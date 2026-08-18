"""candidate_pool.py — candidate pool 구성, distractor 선정 (§4-4)

주의: 후보 풀 크기는 이 파일에 하드코딩하지 않고, 항상
config/sampling_plan.yaml의 candidate_pool_size(§2)에서 읽어온다.
(min/default/max를 두고 min~max 사이에서 랜덤하게 뽑아 문항마다 자연스러운
편차를 준다 — 항상 최솟값으로 고정되는 걸 방지. {"mode": "full_category"}가
주어지면 이 랜덤 샘플링 없이 카테고리 전체를 그대로 pool로 쓴다. {"mode":
"doc_count", "min": ..., "max": ...}가 주어지면 청크 개수가 아니라 문서
개수를 min~max 사이에서 뽑고, 뽑힌 문서들의 청크 전부를 pool로 쓴다.)
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Optional

from query_pipeline import ChunkRef
from text_overlap import overlap_score, tokenize


@dataclass
class CandidateChunk:
    chunk: ChunkRef
    is_positive: bool
    # positive 청크는 항상 None. negative(distractor) 청크만 "hard"/"easy"로 채워진다
    # (§ 질문-청크 키워드 겹침 기반 Hard/Easy Neg 등급, grade_distractors 참고).
    neg_grade: Optional[str] = None


def grade_distractors(query: str, distractors: list[ChunkRef]) -> dict[str, str]:
    """질문과 각 distractor의 키워드 겹침 점수를 매기고, 문항 내 상대 순위로
    상위 절반(점수 > 0인 것만)을 "hard", 나머지를 "easy"로 나눈다.

    distractor 후보 자체(같은 카테고리 다른 문서에서 랜덤 추출, build_candidate_pool
    참고)는 정답 문서와 무관하지만, 그중에서도 질문과 표현이 실제로 겹치는 조항은
    검색기 입장에서 훨씬 헷갈리는 오답이다 — 반대로 같은 문서 안 조항이라도 표현이
    안 겹치면 쉬운 오답일 수 있다. 절대 임계값 대신 문항별 상대 순위(median split)를
    쓰는 이유: 겹침 점수의 절대 분포가 질문마다(길이·용어 등에 따라) 크게 달라서,
    고정 임계값은 문항에 따라 전부 hard이거나 전부 easy로 쏠리기 쉽다."""
    if not distractors:
        return {}
    question_tokens = tokenize(query)
    scored = [(c, overlap_score(question_tokens, c.content)) for c in distractors]
    median = sorted(s for _, s in scored)[len(scored) // 2]
    return {c.chunk_id: ("hard" if s > 0 and s >= median else "easy") for c, s in scored}


def build_candidate_pool(
    data_type: str,
    positive_chunks: list[ChunkRef],
    chunk_pool_for_method: list[ChunkRef],
    pool_size_cfg: dict,
    query: str = "",
    seed: int | None = None,
) -> list[CandidateChunk]:
    """pool_size_cfg: sampling_plan.yaml의 candidate_pool_size[data_type].
    - {"mode": "full_category"}면 cap 없이 카테고리 전체(positive 청크 자신만 제외)를 그대로 pool로 쓴다.
    - {"mode": "doc_count", "min": ..., "max": ...}면 청크 개수가 아니라 문서 개수를 min~max
      사이에서 뽑고, 뽑힌 문서들의 청크를 전부 distractor로 쓴다(문서 단위로 pool 규모를 통제).
    - 그 외(예: {"min": 6, "default": 10, "max": 20})는 기존처럼 min~max 사이에서 랜덤 크기를
      뽑아 그만큼만 distractor로 샘플링한다(하위 호환).
    positive 청크가 속한 문서의 다른 청크(정답이 아닌 조항)도 distractor 후보에 포함한다 —
    정답 청크 자체만 중복 방지로 빼고, 같은 문서의 나머지는 카테고리 전체의 일부로 그대로 쓴다.
    query: distractor의 neg_grade(hard/easy) 판정에 쓰는 문항의 질문 텍스트."""
    rng = random.Random(seed)
    positive_chunk_ids = {c.chunk_id for c in positive_chunks}
    others = [c for c in chunk_pool_for_method if c.chunk_id not in positive_chunk_ids]
    mode = pool_size_cfg.get("mode")

    if mode == "full_category":
        distractors = others
    elif mode == "doc_count":
        doc_ids = list({c.doc_id for c in others})
        rng.shuffle(doc_ids)
        lo, hi = pool_size_cfg["min"], pool_size_cfg["max"]
        if lo > hi:  # config 값이 잘못 들어와도(min>max) 죽지 않도록 방어
            lo, hi = hi, lo
        n_docs = min(rng.randint(lo, hi), len(doc_ids))  # 후보 문서가 부족하면 있는 만큼 전부 사용
        chosen_docs = set(doc_ids[:n_docs])
        distractors = [c for c in others if c.doc_id in chosen_docs]
    else:
        rng.shuffle(others)
        lo, hi = pool_size_cfg["min"], pool_size_cfg["max"]
        if lo > hi:  # config 값이 잘못 들어와도(min>max) 죽지 않도록 방어
            lo, hi = hi, lo
        target_size = rng.randint(lo, hi)  # min~max 사이에서 문항마다 자연스럽게 편차를 둠
        n_distractor = max(target_size - len(positive_chunks), 0)
        n_distractor = min(n_distractor, len(others))  # 후보가 부족하면 있는 만큼 전부 사용
        distractors = others[:n_distractor]

    grades = grade_distractors(query, distractors)
    pool = [CandidateChunk(c, True) for c in positive_chunks] + [
        CandidateChunk(c, False, grades.get(c.chunk_id)) for c in distractors
    ]
    rng.shuffle(pool)
    return pool
