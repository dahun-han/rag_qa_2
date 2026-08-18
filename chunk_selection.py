"""chunk_selection.py — 카테고리/문서 캐싱과 positive 청크 선정 로직.

청킹방식과 무관하게 "어떤 청크를 근거로 쓸지" 결정하는 순수 로직만 모아둔다
(LLM 호출도, 로깅도 없음). main.py는 이 모듈의 함수들을 그대로 재사용한다.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass

from retag import CATEGORY_CODE, RAW_TO_NAME

RAW_NAMES_BY_CODE: dict[str, list[str]] = {}
for raw, name in RAW_TO_NAME.items():
    RAW_NAMES_BY_CODE.setdefault(CATEGORY_CODE[name], []).append(raw)


def build_category_cache(chunk_pool_by_method: dict) -> dict[tuple[str, str], list]:
    """(chunking_method, category_code) -> 후보 청크 리스트를 미리 한 번만 계산해 캐싱한다
    — 문항마다·청킹방식마다 pool 전체를 다시 필터링하면 700건×3방식 규모에서 반복 스캔이 된다."""
    cache: dict[tuple[str, str], list] = {}
    for method, pool in chunk_pool_by_method.items():
        for code, raw_names in RAW_NAMES_BY_CODE.items():
            cache[(method, code)] = [c for c in pool if c.category in raw_names]
    return cache


@dataclass
class DocCandidate:
    doc_id: str
    subcategory: str


def build_document_cache(category_cache: dict[tuple[str, str], list]) -> dict[tuple[str, str], list[DocCandidate]]:
    """(청킹방식, 카테고리 코드) -> 후보 문서 목록(DocCandidate)을 미리 한 번만 계산해
    캐싱한다 — 문서 선택 단계(pick_positive_chunks/pick_excluded_doc)는 이 목록에서 바로
    무작위로 고르기만 하면 된다. 청킹방식마다 독립적으로 질문/답변을 생성하므로(세 방식이
    같은 문서를 가리키더라도) 문서 후보 목록도 그 방식의 청크 풀 기준으로 각각 만든다."""
    cache: dict[tuple[str, str], list[DocCandidate]] = {}
    for (method, code), chunks in category_cache.items():
        seen: dict[str, str] = {}
        for c in chunks:
            seen.setdefault(c.doc_id, c.subcategory)
        cache[(method, code)] = [DocCandidate(doc_id=doc_id, subcategory=sub) for doc_id, sub in seen.items()]
    return cache


def pick_positive_chunks(candidates: list, doc_candidates: list[DocCandidate], n: int, difficulty: str,
                          rng: random.Random, data_type: str = "pos_neg") -> list:
    """카테고리 후보 문서 목록(doc_candidates, build_document_cache로 미리 계산됨)에서
    슬롯마다 다른 문서를 골라 positive 청크를 선정한다. candidates는 그 문서들의 실제
    청크 내용을 가져오는 용도로만 쓴다(문서 선택 자체는 doc_candidates가 담당).
    근거 개수(n)는 3~5로 고정되고, **분산 방식은 difficulty로만 결정**하며
    pos_neg/all_pos가 완전히 동일한 로직을 공유한다:
      하 - 한 문서의 한 위치(인접한 연속 청크들)
      중 - 한 문서 내 여러 위치(서로 다른 topic_list로 흩어진 청크들)
      상 - 두 문서 이상에 근거가 분산
    data_type은 여기서 분산 방식을 바꾸지 않는다 — pos_neg와 all_pos의 차이는
    "최종 positive가 1개로 줄어드는 걸 허용하는가"뿐이고, 그건 이 함수가 아니라
    main.build_item()의 minimal evidence set collapse 검증에서 처리한다 (§4-1/§4-4).
    (rng는 슬롯별로 다른 seed로 생성해서 넘겨야 함 - 안 그러면 항상 같은 문서만 뽑힘)"""
    doc_ids = [d.doc_id for d in doc_candidates]
    rng.shuffle(doc_ids)
    subcategory_by_doc = {d.doc_id: d.subcategory for d in doc_candidates}

    def chunks_of(doc_id: str) -> list:
        return [c for c in candidates if c.doc_id == doc_id]

    def pick_doc_pair() -> tuple[str, str] | None:
        """완전 무작위로 문서 2개를 묶으면 서로 관련 없는 문서끼리 묶여(예: 대여금고약관 +
        외화송금약관) "두 문서를 다 써야만 답할 수 있는" 자연스러운 질문 자체가 존재하기
        어렵다 -> 이게 근거 collapse(한 문서로 좁혀짐, §4-4)의 주된 원인이었다. 같은
        subcategory(예: 둘 다 "담보대출") 문서끼리 우선 짝지어, 비교·결합이 자연스러운
        문서 쌍이 뽑히도록 한다."""
        by_sub: dict[str | None, list[str]] = {}
        for d in doc_ids:
            by_sub.setdefault(subcategory_by_doc.get(d), []).append(d)
        same_sub_groups = [ids for ids in by_sub.values() if len(ids) >= 2]
        if same_sub_groups:
            group = rng.choice(same_sub_groups)
            rng.shuffle(group)
            return group[0], group[1]
        if len(doc_ids) >= 2:  # 같은 subcategory 쌍이 없으면 무작위 폴백
            return doc_ids[0], doc_ids[1]
        return None

    if difficulty == "상" and len(doc_ids) > 1:
        pair = pick_doc_pair()
        if pair:
            doc1, doc2 = pair
            first_half = max(n // 2, 1)
            first_doc_chunks = chunks_of(doc1)
            second_doc_chunks = chunks_of(doc2)
            rng.shuffle(first_doc_chunks)
            rng.shuffle(second_doc_chunks)
            chosen = first_doc_chunks[:first_half] + second_doc_chunks[: n - first_half]
            if chosen:
                return chosen
        # 두 번째 문서에서 하나도 못 뽑았으면(예외적 상황) 아래 폴백으로 진행

    doc_chunks = chunks_of(doc_ids[0])
    if not doc_chunks:
        return candidates[:n]

    if difficulty == "중" and len(doc_chunks) > n:
        # 서로 다른 topic(조항)으로 흩어진 청크를 우선 선택 -> 조건 조합이 필요하게 만듦
        by_topic: dict[tuple, list] = {}
        for c in doc_chunks:
            key = tuple(c.topic_list) or (c.chunk_id,)
            by_topic.setdefault(key, []).append(c)
        topic_groups = list(by_topic.values())
        rng.shuffle(topic_groups)
        for g in topic_groups:
            rng.shuffle(g)
        chosen: list = []
        gi = 0
        while len(chosen) < n and any(topic_groups):
            group = topic_groups[gi % len(topic_groups)]
            if group:
                chosen.append(group.pop(0))
            gi += 1
        if len(chosen) >= min(n, 2):  # 서로 다른 topic이 최소 2개는 섞였을 때만 채택
            return chosen[:n]

    # 하 (또는 위 조건에 해당 안 되는 fallback): 문서 내 인접한(연속된) 청크 구간을 그대로 사용
    doc_chunks_sorted = sorted(doc_chunks, key=lambda c: c.start_char)
    if len(doc_chunks_sorted) <= n:
        return doc_chunks_sorted
    start = rng.randint(0, len(doc_chunks_sorted) - n)
    return doc_chunks_sorted[start : start + n]


def pick_excluded_doc(candidates: list, doc_candidates: list[DocCandidate], rng: random.Random):
    """all_neg용: 후보 문서 목록(doc_candidates) 중 슬롯마다 다른 문서를 '실제로는
    정답이 있지만 제외된' 문서로 고른다."""
    chosen_doc = rng.choice([d.doc_id for d in doc_candidates])
    return [c for c in candidates if c.doc_id == chosen_doc]


def prune_to_minimal_evidence(chunks: list, nuggets: list) -> list:
    """positive 후보(chunks) 중 실제로 어떤 nugget이라도 뒷받침하는 청크만 남긴다
    (§4-4 minimal evidence set). 나머지(질문 생성용으로만 쓰이고 답변엔 안 쓰인
    이웃 청크 등)는 제거해 Precision을 끌어올린다. 전부 제거되면(추출 실패 등)
    안전하게 원본을 그대로 반환한다."""
    if not chunks or not nuggets:
        return chunks
    kept = []
    for ch in chunks:
        content_norm = re.sub(r"\s+", "", ch.content)
        supports_any = any(
            re.sub(r"\s+", "", str(n.value)) and re.sub(r"\s+", "", str(n.value)) in content_norm
            for n in nuggets
            if n.key != "근거 존재 여부"
        )
        if supports_any:
            kept.append(ch)
    return kept or chunks


def classify_difficulty(positive_chunks: list, fallback_difficulty: str) -> tuple[str, int, int]:
    """positive_chunks의 실제 조항(topic) 수·문서 수를 세어 난이도(하/중/상)를 정직하게
    재계산한다(§3-3). "청크 개수"가 아니라 topic_list 기준 조항 수/문서 수로 판정하며,
    positive_chunks가 비어 있으면(all_neg) 재계산할 근거가 없으므로 fallback_difficulty를
    그대로 유지한다."""
    distinct_topics = {tuple(c.topic_list) for c in positive_chunks if c.topic_list}
    n_topics = len(distinct_topics) if distinct_topics else (1 if positive_chunks else 0)
    n_docs = len({c.doc_id for c in positive_chunks})

    if n_docs >= 2:
        difficulty = "상"
    elif n_topics >= 2:
        difficulty = "중"
    elif positive_chunks:
        difficulty = "하"
    else:
        difficulty = fallback_difficulty
    return difficulty, n_topics, n_docs
