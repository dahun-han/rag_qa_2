"""validation.py — QA 품질 검증(§4-6) + 구조적 정합성 검증 (필수 게이트)"""

from __future__ import annotations

import re

from answer_gen import ALL_NEG_ANSWER, Nugget
from candidate_pool import CandidateChunk
from query_pipeline import rule_based_filter
from query_pipeline import _VAGUE_REF_RE  # GT의 지시어 남용 여부도 같은 정규식으로 체크

_TOKEN_RE = re.compile(r"[가-힣A-Za-z0-9]+")


def _value_supported(value: str, ground_truth: str, min_overlap: float = 0.5) -> bool:
    """nugget value가 ground_truth로 뒷받침되는지 판정.
    완전 문자열 일치를 요구하면 요약/재구성된 value(정상적인 경우도 많음)가 계속
    걸리므로, 정규화 후 부분일치 -> 토큰별 substring 포함 비율(min_overlap 이상) 순으로
    완화 판정한다. (한국어는 조사가 붙어 토큰이 정확히 안 맞는 경우가 많아, set 교집합이
    아니라 각 토큰이 정규화된 ground_truth 문자열에 부분 포함되는지로 본다.)"""
    value_norm = re.sub(r"\s+", "", str(value))
    gt_norm = re.sub(r"\s+", "", ground_truth)
    if not value_norm or value_norm in gt_norm:
        return True

    value_tokens = _TOKEN_RE.findall(str(value))
    if not value_tokens:
        return True
    matched = sum(1 for t in value_tokens if t in gt_norm)
    return (matched / len(value_tokens)) >= min_overlap


_META_PHRASES = ["발췌문에 따르면", "발췌문에 의하면", "제공된 자료에", "문서에 따르면", "제공된 내용에"]


def validate_query_answer(
    query: str,
    ground_truth: str,
    positive_chunk_ids: list[str],
    nuggets: list[Nugget],
    reason: str,
    grounded: bool,
    source_text: str = "",
) -> tuple[bool, list[str]]:
    """§4-6: 문항 내용 차원의 필수 검증 (7항목 중 코드로 자동화 가능한 항목).
    source_text: nugget 추출에 쓰인 원본 청크(들)의 content를 이어붙인 문자열.
    nugget value는 ground_truth가 아니라 이 원본 텍스트에서 검증한다
    (nugget이 존댓말로 다듬어진 답변이 아니라 원문 그대로여야 하므로)."""
    failures: list[str] = []

    ok, why = rule_based_filter(query)
    if not ok:
        failures.append(f"질문 발화 형식: {why}")

    if not grounded:
        failures.append("답변 grounding 실패 (positive 근거 밖 내용 포함 의심)")

    if _VAGUE_REF_RE.search(ground_truth) and not re.search(
        r"[가-힣A-Za-z0-9]{2,}(설명서|약관|안내|규정|서비스|상품|계좌|예금|대출|서식|신청서|계약서|확인서|명세서)",
        ground_truth,
    ):
        failures.append("답변에 '이 서식/이 문서' 등 지시어만 있고 실제 상품명·문서명이 없음")

    if ground_truth != ALL_NEG_ANSWER:
        for phrase in _META_PHRASES:
            if phrase in ground_truth:
                failures.append(f"답변에 메타 표현 '{phrase}' 포함 (바로 사실만 서술해야 함)")

    for cid in positive_chunk_ids:
        if cid not in reason:
            failures.append(f"reason에 positive_chunk_ids({cid}) 누락")

    check_target = source_text or ground_truth  # source_text 없으면(all_neg 등) GT로 폴백
    for n in nuggets:
        if n.key == "근거 존재 여부":
            continue
        if not _value_supported(n.value, check_target):
            failures.append(f"nugget '{n.key}'의 value가 원본 발췌문에서 확인 안 됨")

    return len(failures) == 0, failures


def validate_structure(
    positive_chunk_ids: list[str],
    candidate_pool: list[CandidateChunk],
    positive_docs: dict,
) -> tuple[bool, list[str]]:
    """구조적 정합성 검증 (schema/consistency, 청킹방식별 재실행 대상)."""
    failures: list[str] = []
    candidate_ids = {cc.chunk.chunk_id for cc in candidate_pool}

    if not set(positive_chunk_ids).issubset(candidate_ids):
        failures.append("positive_chunk_ids가 candidate_chunk_ids의 부분집합이 아님")

    positive_doc_ids = {cc.chunk.doc_id for cc in candidate_pool if cc.is_positive}
    if positive_doc_ids != set(positive_docs.keys()):
        failures.append("positive_docs의 key 집합이 실제 positive 청크의 문서집합과 불일치")

    return len(failures) == 0, failures
