"""answer_gen.py — Gold 답변 생성(§4-3), grounding 검증, Nugget 추출"""

from __future__ import annotations

from dataclasses import dataclass

from query_pipeline import ChunkRef, LLMClient
from timing import timed

ALL_NEG_ANSWER = "문의하신 내용은 현재 확인 가능한 범위에서 찾을 수 없습니다."


@dataclass
class Nugget:
    id: str
    key: str
    value: str
    type: str  # enum|bool|numeric|text
    vital: bool


_ANSWER_SYSTEM = """주어진 발췌문에 명시된 내용만 사용해 질문에 답하라.
발췌문에 없는 내용은 절대 추측하거나 채워넣지 말라. 발췌문에 없으면
'명시되어 있지 않음'이라고 답하라.
단, 발췌문이 명시한 정의나 절차로부터 질문의 답이 논리적으로 바로 도출되는 경우
(새로운 사실을 끌어오지 않고 이미 쓰인 내용을 그대로 적용하기만 하면 되는 경우)에는
그 결론을 답변에 포함하라 — 이런 경우까지 '명시되어 있지 않음'으로 답하면 안 된다.
예를 들어 발췌문이 "일괄계좌신규는 사업주가 실명확인한 서류로 일괄 처리하는 것"이라고
정의한다면 "그러므로 고객이 별도로 방문해 신분 확인을 받을 필요는 없다"는 결론을 답변에
써도 된다. 반대로 발췌문의 정의·절차만으로는 유추되지 않고 별도로 확인돼야 하는 사실
(예: 수수료, 예외 조건, 추가 서류 등)은 여전히 '명시되어 있지 않음'이라고 답하라.
2~4문장의 자연스러운 서술형으로 답하라.
"발췌문에 따르면", "제공된 자료에 의하면", "문서에 따르면" 같은 메타 표현은 쓰지 말고,
실제 상담/안내 답변처럼 바로 사실을 서술하라.
반드시 정중한 존댓말(합니다/됩니다 체)로 작성한다. 은행 상담원이 고객에게 안내하는
느낌으로, 딱딱한 사무 문서 말투가 아니라 예의 바르고 친절한 어투를 쓴다.
문서를 지칭할 때도 "이 서식은", "이 문서에서는" 같은 지시어만 쓰지 말고,
"OO대출 서식에서는", "예금거래용 CD수익률 설명서에서는"처럼 실제 상품·서식명을
자연스럽게 넣어 무엇에 대한 안내인지 명확히 한다."""

_ANALYSIS_SYSTEM = """발췌문과 답변을 보고 아래 세 가지를 한 번에 판정해 JSON으로만 답하라.
발췌문은 [chunk_id=...] 라벨이 붙은 여러 조각으로 주어진다.

{
  "grounded": true|false,
  "unsupported_sentence": "grounded가 false일 때만: 근거 없는 문장",
  "essential_chunk_ids": ["실제로 이 답변을 만드는 데 반드시 필요했던 chunk_id만"],
  "nuggets": [{"id": "N1", "key": "...", "value": "...", "type": "enum|bool|numeric|text", "vital": true|false}, ...]
}

- grounded: 답변의 모든 문장이 발췌문 내용만으로 뒷받침되는가. 발췌문에 명시된 정의·절차에서
  논리적으로 바로 도출되는 결론 문장도 뒷받침되는 것으로 본다 — 발췌문에 없는 새로운 사실을
  끌어온 문장이 있을 때만 grounded=false로 판정한다
- essential_chunk_ids: 발췌문에는 여러 chunk_id가 섞여 있지만, 그중 답변을 만드는 데
  **실제로 필요했던 chunk_id만** 골라라. 답변과 무관하거나 곁다리로만 스쳐 지나간
  chunk_id는 포함하지 마라. 최소한으로, 정말 없으면 답을 만들 수 없었던 것만 남긴다.
- nuggets: 답변을 "왜 그런지 설명하는 서술문(reason)"이 아니라, 문서에 명시된 개별
  사실을 항목별로 뽑아낸 체크리스트로 만든다. nugget 1개 = 사실 1개. 여러 사실을
  한 nugget에 섞어 넣지 말고, 답변에 담긴 사실 개수만큼 nugget을 쪼갠다.
  key는 그 사실이 속한 대분류 항목명(한국어 명사구, 영어 금지)이다. **지금보다 더
  넓은 상위 주제 단위로 잡아라** — 사실 하나하나마다 전용 key를 새로 만들지 말고,
  같은 화제(예: 해지 절차, 수수료 부과 기준, 심사 요건)에 속한 여러 nugget이 있으면
  같은 key를 공유하도록 상위 개념으로 묶어라. key가 사실상 그 nugget 하나만을
  가리키는 이름이 되어버리면(예: 사실 자체를 그대로 옮긴 "열쇠 반납 시점") 너무
  좁은 것이다 — 그 정도로 좁은 구분은 key가 아니라 value가 담당한다.
  **value는 그 대분류(key) 아래 실제 개별 사실(누가/언제/얼마/무엇)을 짧게 담는다.**
  문장 전체를 옮기지 말고, 발췌문에 실제로 쓰인 표현에서 핵심 명사구만 뽑아써라.
  "~로 간주합니다", "~하여야 한다", "~됩니다"처럼 서술어로 끝나는 완결된 문장을
  통째로 옮기는 것은 금지한다. 단, 새 단어를 창작하거나 요약해서 바꿔 쓰지 말고
  발췌문의 표현을 그대로 잘라써야 한다(값이 재구성되면 "답변에 이 표현이 있는가"라는
  채점 기준 자체가 흔들린다).

  - 나쁜 예(key가 nugget마다 제각각이라 사실상 소분류): key: "해지 신청 주체" /
    value: "임차인", key: "사전 통지" / value: "1개월 전, 서면 또는 전화", key:
    "열쇠 반납 시점" / value: "해지 시" — 세 key가 전부 다른 개념처럼 갈라져 있지만
    실제로는 전부 "해지 절차"라는 같은 상위 주제에 속한 사실들이다.
  - 좋은 예(같은 상위 주제는 key를 공유하고, value가 개별 사실을 구분): key: "해지
    절차" / value: "임차인이 해지 신청", key: "해지 절차" / value: "1개월 전 서면
    또는 전화로 사전 통지", key: "해지 절차" / value: "해지 시 열쇠 반납"

[형식 예시 — 서로 다른 상위 주제(대분류)는 다른 key로, 같은 주제에 속한 사실은 value로만 구분한다]
{"nuggets": [
  {"id": "N1", "key": "해지 절차", "value": "임차인이 해지 신청", "type": "enum", "vital": true},
  {"id": "N2", "key": "해지 절차", "value": "1개월 전 서면 또는 전화로 사전 통지", "type": "text", "vital": true},
  {"id": "N3", "key": "해지 절차", "value": "해지 시 열쇠 반납", "type": "text", "vital": true}
]}"""


def generate_ground_truth(
    llm: LLMClient, query: str, positive_chunks: list[ChunkRef], reasoning_effort: str | None = None,
) -> str:
    if not positive_chunks:  # all_neg
        return ALL_NEG_ANSWER
    excerpt = "\n\n".join(c.content for c in positive_chunks)
    with timed("GT"):
        return llm.complete(
            _ANSWER_SYSTEM, f"[발췌문]\n{excerpt}\n\n[질문]\n{query}",
            temperature=0.0, reasoning_effort=reasoning_effort,
        )


def analyze_answer(
    llm: LLMClient, ground_truth: str, positive_chunks: list[ChunkRef], reasoning_effort: str | None = None,
) -> tuple[bool, list[Nugget], list[str]]:
    """grounding 체크 + nugget 추출 + 최소 필수근거(essential_chunk_ids) 판정을
    API 호출 1번으로 처리한다. 반환: (grounded, nuggets, essential_chunk_ids)."""
    if not positive_chunks or ground_truth == ALL_NEG_ANSWER:
        return True, [Nugget(id="N1", key="근거 존재 여부", value="없음", type="bool", vital=True)], []

    excerpt = "\n\n".join(f"[chunk_id={c.chunk_id}]\n{c.content}" for c in positive_chunks)
    with timed("Analyze"):
        result = llm.complete_json(
            _ANALYSIS_SYSTEM, f"[발췌문]\n{excerpt}\n\n[답변]\n{ground_truth}", reasoning_effort=reasoning_effort,
        )

    grounded = bool(result.get("grounded"))
    nuggets = [
        Nugget(id=f"N{i}", key=n["key"], value=n["value"], type=n.get("type", "text"), vital=bool(n.get("vital", True)))
        for i, n in enumerate(result.get("nuggets", []), start=1)
    ]
    valid_ids = {c.chunk_id for c in positive_chunks}
    essential_ids = [cid for cid in result.get("essential_chunk_ids", []) if cid in valid_ids]
    return grounded, nuggets, essential_ids
