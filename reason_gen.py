"""reason_gen.py — 논리적 서술형 reason 생성 (§4-5), 한 문장으로 압축"""

from __future__ import annotations

import re

from query_pipeline import LLMClient
from timing import timed

# chunk_pool.py가 넘기는 doc_nm/excluded_doc_nm은 원본 입력의 source_doc 값을 그대로
# 쓴다("수신__계좌관리__16cd108821e53b60_계좌이동서비스_약관"처럼 카테고리·소분류·
# 문서ID가 밑줄로 뭉쳐진 원본 파일명) — 이걸 그대로 reason 프롬프트에 "문서명"으로
# 넘기면 모델이 그 지저분한 원본 표기를 그대로 베껴 쓴다. 여기서 카테고리/소분류/
# 문서ID 접두사를 떼고 밑줄을 공백으로 바꿔, 사람이 읽을 문서명만 남긴다.
_DOC_ID_PREFIX_RE = re.compile(r"^[0-9a-f]{8,}_")


def clean_doc_name(raw: str) -> str:
    parts = raw.split("__")
    tail = parts[-1] if len(parts) >= 3 else raw
    tail = _DOC_ID_PREFIX_RE.sub("", tail)
    return tail.replace("_", " ").strip() or raw

_REASON_SYSTEM = """다음 정보를 바탕으로, 이 문항이 왜 이렇게 만들어졌는지 한 문장으로 서술하라.
라벨이나 번호는 붙이지 말고, 아래 세 가지를 한 문장 안에 자연스러운 절로 엮어 써라:
(1) 질문이 어떤 내용을 근거로 도출됐는지, (2) 답변(GT)이 어느 근거 청크 id를
바탕으로(distractor 없이) 작성되었는지, (3) 근거가 몇 개의 조항(청크 개수가 아니라
실제로 서로 다른 조항·주제의 개수)으로 어떻게 분산돼 있어 난이도 [하/중/상]으로
선정되었는지.

**절대 규칙: 대괄호(`[`, `]`)나 따옴표로 감싼 리스트 형태를 단 하나도 쓰지 마라.**
근거 청크 id나 문서명이 여러 개여도 목록을 그대로 나열하지 말고, 완전한 한국어
문장 성분으로 풀어써라. 근거 청크 id 값은 문장 어딘가에 자연스러운 문장 성분
(주어/목적어/수식어)으로 반드시 그대로 포함시켜라.

- 나쁜 예(절대 금지): "['a3f96723ccb7faed_42', 'a3f96723ccb7faed_43']에 근거하여"
- 좋은 예(한 문장 통합): "질문은 신한저축은행 연계대출(허그론) 대출약정 통합서식의
  중도상환 조건을 근거로 만들어졌고, 답변은 근거 청크 a3f96723ccb7faed_42의
  내용을 바탕으로 작성되었으며, 근거 1개가 단일 위치에 모여있어 난이도 '하'로
  선정되었다.\""""


def generate_reason(llm: LLMClient, ctx: dict) -> str:
    """
    ctx keys: query, ground_truth, positive_chunk_ids, doc_names, data_type,
              difficulty, n_positive, n_topics, doc_spread(str), excluded_doc_nm(옵션)
    n_topics: 실제 근거의 서로 다른 조항(topic) 개수 — 난이도 판정 기준이 청크
    개수가 아니라 이 값이라는 걸 reason 서술에서도 일관되게 반영하기 위해 넘긴다.
    """
    # id/문서명을 파이썬 리스트 그대로(['a','b']) 보여주면 모델이 그 대괄호 표기를
    # 그대로 따라 쓰는 경향이 있었다. 쉼표로만 구분한 평범한 문자열로 바꿔서,
    # 입력 단계에서부터 리스트 표기를 볼 일이 없게 한다.
    ids_str = ", ".join(ctx["positive_chunk_ids"]) or "(없음)"
    docs_str = ", ".join(clean_doc_name(d) for d in ctx["doc_names"]) or "(없음)"
    user = (
        f"질문: {ctx['query']}\n답변: {ctx['ground_truth']}\n"
        f"근거 청크 id(쉼표로만 구분됨, 대괄호·따옴표 없이 문장에 자연스럽게 녹여 쓸 것): {ids_str}\n"
        f"문서명(쉼표로만 구분됨): {docs_str}\n"
        f"data_type: {ctx['data_type']} / difficulty: {ctx['difficulty']}\n"
        f"근거 조항 수(청크 개수 아님, 실제 서로 다른 조항·주제 개수): {ctx.get('n_topics', ctx['n_positive'])}\n"
        f"근거 분산: {ctx['doc_spread']}"
    )
    if ctx.get("excluded_doc_nm"):
        user += f"\n(all_neg) 실제 정답이 있으나 제외된 문서: {clean_doc_name(ctx['excluded_doc_nm'])}"

    # 이미 확정된 사실(질문/답변/근거 id)을 정해진 템플릿으로 서술만 하는 기계적
    # 작업이라 낮은 추론 강도로도 충분함 — judge/persona 분류 호출과 같은 이유로
    # reasoning_effort="low" 적용(비용 절감). generate_ground_truth는 실제 답변
    # 내용을 새로 합성해야 해서 여기 포함하지 않음.
    with timed("Reason"):
        reason = llm.complete(_REASON_SYSTEM, user, temperature=0.3, reasoning_effort="low")

    # positive_chunk_ids 누락 검증 (§4-5 필수 조건) - 1문장 형식을 유지하기 위해
    # 줄바꿈 없이 문장 끝에 이어붙인다.
    for cid in ctx["positive_chunk_ids"]:
        if cid not in reason:
            reason += f" (근거 청크: {ids_str})"
            break
    return reason
