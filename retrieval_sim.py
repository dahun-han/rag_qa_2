"""retrieval_sim.py — 키워드 겹침 기반 저비용 모의 검색 (임베딩 API 미사용).

retrieval.json/answer.json을 만들 때 문항의 data_type을 "실제로 이 질문이 이
검색 방식으로 얼마나 잘 풀리는가"로 재분류하고, answer.json의 context를 이 모의
검색의 top_k 결과 그대로 채우는 데 쓴다. 형태소 분석기는 쓰지 않는 러프한 1차
구현 — 필요하면 나중에 정교한 검색기로 교체 가능하도록 candidate_pool.py.CandidateChunk만
의존한다.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from candidate_pool import CandidateChunk
from text_overlap import overlap_score, tokenize


@dataclass
class RetrievalSim:
    top_k: int
    topk_chunks: list[CandidateChunk]   # 점수 내림차순, 길이 top_k(또는 pool이 더 작으면 그만큼)
    n_pos: int
    data_type: str                      # 재분류된 값: all_pos/pos_neg/all_neg


def simulate_retrieval(
    index: str,
    query: str,
    candidate_pool: list[CandidateChunk],
    top_k_min: int,
    top_k_max: int,
) -> RetrievalSim:
    """문항 index로 시드를 고정해 top_k를 뽑고, candidate_pool을 키워드 겹침
    점수로 정렬해 상위 top_k를 추출한다. 그 안의 실제 positive 개수로 data_type을
    재분류한다. topk_chunks 자체가 answer.json의 context로 그대로 쓰인다(positive와
    neg를 실제로 top_k 안에 뽑힌 만큼만 담음 — GOLD positive_chunk_ids 전체가 아님)."""
    rng = random.Random(hash(index) % (2**31))
    top_k = rng.randint(top_k_min, top_k_max)
    depth = min(top_k, len(candidate_pool))

    question_tokens = tokenize(query)
    pool = list(candidate_pool)
    rng.shuffle(pool)  # 동점 처리: 점수가 같으면 이 셔플 순서로 안정적으로 갈린다
    pool.sort(key=lambda cc: overlap_score(question_tokens, cc.chunk.content), reverse=True)
    topk_chunks = pool[:depth]

    n_pos = sum(1 for cc in topk_chunks if cc.is_positive)
    if n_pos == depth and depth > 0:
        data_type = "all_pos"
    elif n_pos == 0:
        data_type = "all_neg"
    else:
        data_type = "pos_neg"

    return RetrievalSim(top_k=top_k, topk_chunks=topk_chunks, n_pos=n_pos, data_type=data_type)
