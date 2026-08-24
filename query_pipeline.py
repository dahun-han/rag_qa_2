"""
query_pipeline.py — Q0~Q8 질의 생성 파이프라인 (OpenAI, 모델은 config/llm_config.yaml에서 설정)
"""

from __future__ import annotations

import json
import logging
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Literal, Optional, Protocol

from prompts import (
    _BAD_EXAMPLES,
    _BATCH_JUDGE_SYSTEM,
    _BATCH_QUERY_ATTRS_SYSTEM,
    _GOOD_EXAMPLES,
    _JUDGE_SYSTEM,
    _MULTIHOP_ADDENDUM,
    _QUERY_ATTRS_SYSTEM,
    _QUERY_STYLE_GUIDE,
    _REASONING_ADDENDUM,
    _TERM_AVOIDANCE_ADDENDUM,
)
from timing import timed

logger = logging.getLogger("query_pipeline")
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
# openai/httpx의 "HTTP Request: POST ... 200 OK" 같은 진행 로그를 끈다.
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

DataType = Literal["pos_neg", "all_pos", "all_neg"]
Difficulty = Literal["하", "중", "상"]
Persona = Literal["고객", "내부직원"]
QueryStyle = Literal["정보형", "확인형", "짧은검색형", "문제해결형", "절차문의형", "불완전형"]

@dataclass
class ChunkRef:
    chunk_id: str
    doc_id: str
    doc_nm: str
    category: str
    subcategory: str
    content: str
    topic_list: list[str] = field(default_factory=list)
    start_char: int = 0
    end_char: int = 0


@dataclass
class QuerySlot:
    index: str
    data_type: DataType
    difficulty: Difficulty
    positive_chunks: list[ChunkRef]
    target_persona: Optional[Persona] = None
    excluded_doc_nm: Optional[str] = None
    # 질문 생성 시 실제로 보여줄 발췌문. all_neg처럼 positive_chunks가 비어있어도
    # (실제로는 정답이 있지만 후보군에서 제외된) 문서 내용을 보고 질문을 만들어야
    # 하므로, 이 필드가 있으면 이걸 우선 사용한다. 없으면 positive_chunks를 사용.
    context_chunks: list[ChunkRef] = field(default_factory=list)

    def context(self) -> list[ChunkRef]:
        return self.context_chunks or self.positive_chunks


@dataclass
class QueryResult:
    query: str
    persona: Persona
    focus_topic: str
    attempts: int
    passed: bool
    query_style: Optional[QueryStyle] = None  # 사후 분류 기록용 — target 아님, 재시도 트리거 안 함
    fail_reason: Optional[str] = None


@dataclass
class QueryGenConfig:
    max_attempts: int = 3
    temperature: float = 0.7
    batch_size: int = 3  # 한 라운드에 후보 질문을 몇 개씩 만들어 한꺼번에 검증할지
    max_tokens: Optional[int] = None  # 배치 생성 콜 전용 max_tokens override (None=전역 설정 사용).
                                       # batch_size가 클수록 추론+출력 토큰이 많이 필요해 전역
                                       # max_tokens로는 "빈 응답(finish_reason=length)"이 잦았다.


# LLM 클라이언트
class LLMClient(Protocol):
    def complete(
        self, system: str, user: str, temperature: float = 0.7,
        reasoning_effort: Optional[str] = None, max_tokens: Optional[int] = None,
    ) -> str: ...
    def complete_json(
        self, system: str, user: str, reasoning_effort: Optional[str] = None, max_tokens: Optional[int] = None,
    ) -> dict: ...


class OpenAILLMClient:
    """pip install openai, 환경변수 OPENAI_API_KEY 필요.
    모델/파라미터는 config/llm_config.yaml(llm_config.load_llm_config)로 관리한다."""

    def __init__(self, cfg=None, model: str = "gpt-5-mini", max_retries: int = 3, retry_backoff: float = 2.0):
        from openai import OpenAI
        self._client = OpenAI()
        if cfg is None:
            from llm_config import LLMConfig
            cfg = LLMConfig(model=model)
        self._cfg = cfg
        self._max_retries = max_retries
        self._retry_backoff = retry_backoff

    def _base_kwargs(self, reasoning_effort: Optional[str] = None, max_tokens: Optional[int] = None) -> dict:
        kwargs = {"model": self._cfg.model}
        if self._cfg.supports_temperature:
            kwargs["temperature"] = self._cfg.temperature
        # 콜별로 max_tokens를 따로 줄 수 있게 한다 — 배치로 여러 후보를 한 번에 써내야
        # 하는 콜(예: 질문 후보 생성)은 추론+출력 토큰이 훨씬 많이 필요해서, 전역
        # max_tokens로는 자주 "빈 응답(finish_reason=length)"이 나 재시도만 반복하다
        # 끝났다 — 인자로 넘긴 값이 있으면 그걸 우선 쓴다.
        effective_max_tokens = max_tokens if max_tokens is not None else self._cfg.max_tokens
        if effective_max_tokens:
            # gpt-5-mini 등 추론 모델은 max_tokens가 아니라 max_completion_tokens를 쓴다.
            # (추론 토큰 + 실제 답변 토큰을 합친 한도이므로 너무 낮게 잡으면 빈 응답이 나올 수 있다.)
            kwargs["max_completion_tokens"] = effective_max_tokens
        if reasoning_effort:
            # persona 분류·judge 예/아니오 판정처럼 단순한 호출은 ground_truth/nugget
            # 생성과 같은 무거운 추론이 필요 없다 — 이런 호출에만 낮은 effort를 넘겨
            # 불필요한 추론 토큰 소비(비용)를 줄인다.
            kwargs["reasoning_effort"] = reasoning_effort
        return kwargs

    def _call_with_retry(
        self, validate=None, reasoning_effort: Optional[str] = None, max_tokens: Optional[int] = None, **kwargs
    ) -> str:
        """API 호출 + 응답 검증을 재시도(지수 백오프)와 함께 수행.
        빈 응답(추론 토큰만 쓰고 실제 답변이 안 나온 경우 등)이나 validate 실패 시 재시도한다."""
        last_err: Exception | None = None
        for attempt in range(1, self._max_retries + 1):
            try:
                resp = self._client.chat.completions.create(
                    **self._base_kwargs(reasoning_effort, max_tokens), **kwargs
                )
                choice = resp.choices[0]
                content = choice.message.content
                if not content:
                    raise ValueError(f"빈 응답 (finish_reason={choice.finish_reason})")
                if validate:
                    validate(content)
                return content
            except Exception as e:  # noqa: BLE001 - 네트워크/파싱 등 다양한 에러를 모두 재시도 대상으로 취급
                last_err = e
                if attempt < self._max_retries:
                    time.sleep(self._retry_backoff * attempt)
        raise RuntimeError(f"OpenAI 호출 {self._max_retries}회 재시도 후에도 실패: {last_err}") from last_err

    def complete(
        self, system: str, user: str, temperature: float = 0.7,
        reasoning_effort: Optional[str] = None, max_tokens: Optional[int] = None,
    ) -> str:
        content = self._call_with_retry(
            reasoning_effort=reasoning_effort, max_tokens=max_tokens, messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return content.strip()

    def complete_json(
        self, system: str, user: str, reasoning_effort: Optional[str] = None, max_tokens: Optional[int] = None,
    ) -> dict:
        content = self._call_with_retry(
            validate=json.loads,  # 파싱 안 되면 재시도 트리거
            reasoning_effort=reasoning_effort,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return json.loads(content)


class MockLLMClient:
    """API 없이 로직만 검증할 때 사용."""

    def __init__(self, canned_queries: Optional[list[str]] = None):
        self._queue = list(canned_queries or [])
        self._default = "정기예금을 6개월 만에 해지하면 이율은 어떻게 적용되나요?"

    def complete(
        self, system: str, user: str, temperature: float = 0.7, reasoning_effort: Optional[str] = None
    ) -> str:
        return self._queue.pop(0) if self._queue else self._default

    def complete_json(self, system: str, user: str, reasoning_effort: Optional[str] = None) -> dict:
        if "persona" in system.lower():
            return {"persona": "고객", "query_style": "정보형", "reason": "1인칭 상담 어투 사용"}
        if '"candidates"' in system:  # 배치 질의 생성 요청
            n = 1
            m = re.search(r"배열 길이 정확히 (\d+)", system)
            if m:
                n = int(m.group(1))
            base = self._queue.pop(0) if self._queue else self._default
            return {"candidates": [base] * n}
        if '"results"' in system:  # 배치 judge 요청
            n = len(re.findall(r"^\d+\.", user, flags=re.MULTILINE)) or 1
            return {"results": [{"answerable_from_context": True, "single_natural_intent": True,
                                  "self_contained": True, "open_ended": True,
                                  "terms_paraphrased": True, "reason": "ok"}] * n}
        return {"answerable_from_context": True, "single_natural_intent": True,
                "self_contained": True, "open_ended": True, "terms_paraphrased": True, "reason": "ok"}


# Q1 — 소재 선정
_EXCLUDED_TOPIC_RE = re.compile(r"약관|권리|정보제공|동의")


def select_focus_topic(slot: QuerySlot) -> str:
    topics: list[str] = []
    for ch in slot.context():
        topics.extend(ch.topic_list)
    topics = [t for t in dict.fromkeys(topics) if t]
    # 약관 시행일·제공절차, 권리·의무 고지, 정보제공/동의 절차 같은 소재는 상품마다
    # 거의 똑같은 정형 문구(법정 필수 고지사항)라 질문 변별력이 없다 — 그 문서만의
    # 고유한 사실이 아니라 어느 문서를 근거로 답하든 비슷한 답이 나온다. 질문 소재로
    # 아예 고려하지 않는다 — 같은 청크에 다른 소재가 있으면 그쪽을 쓴다.
    topics = [t for t in topics if not _EXCLUDED_TOPIC_RE.search(t)]

    if not topics:
        return "해당 문서의 핵심 내용"
    if slot.data_type == "all_pos":
        # rule_based_filter가 가운뎃점(·)을 문어체 나열 기호로 금지하므로, 여기서
        # 프롬프트에 보여줄 소재 목록에도 그 기호를 쓰지 않는다 - 모델이 방금 본
        # 구분자를 답습해 생성 질문에도 "·"를 그대로 쓰는 경향이 있었다.
        return ", ".join(topics[:3])
    return topics[0]


# Q2 — 프롬프트 구성
def build_system_prompt(slot: QuerySlot, batch_size: int = 1) -> str:
    data_type_instruction = {
        "pos_neg": "질문의 답은 근거 안의 특정 사실 하나로 수렴해야 하지만, 그 사실에 이르는 과정은 "
        "규칙 11의 추론(사실 결합/상황 대입/비교) 중 하나를 실제로 거쳐야 한다 — 발췌문 문장을 그대로 "
        "옮겨 묻는 단순 확인형은 여기서도 금지다(규칙 11과 동일 기준). 상황 대입은 어렵게 만들 필요 "
        "없다 — 발췌문의 일반 규정(금지·정의·요건)을 발췌문에 없는 구체적 인물·행동으로 바꿔 "
        "\"이 경우도 해당되나요?\"라고 묻는 정도면 충분하다(예: 발췌문이 \"타인의 실명으로 거래 "
        "금지\"라고만 하면 -> \"제가 배우자 명의로 거래해도 해당되나요?\"). "
        "\"차명거래하면 처벌은 어떻게 되나요?\"처럼 발췌문 문장을 그대로 옮겨 묻기만 하는 건 금지.",
        "all_pos": "질문은 하나의 상황·의도 안에서 자연스럽게 여러 조건을 잇는 형태로 만든다. 나열형은 금지.",
        "all_neg": "질문은 주어진 내용을 정확히 묻는 형태로 만들되, 발화체 1문장·1질문을 지킨다.",
    }[slot.data_type]

    yes_no_note = (
        "   **예/아니오로 답이 갈리는 폐쇄형 질문(\"~되나요?\", \"~가능한가요?\", "
        "\"~있나요?\")은 금지는 아니지만 최대한 줄이고**, \"얼마나/어떻게/언제/무엇을\" "
        "같은 개방형을 우선한다."
    )
    if batch_size > 1:
        yes_no_note += f" 이번 {batch_size}개 후보 중 예/아니오형은 최대 1개까지만."

    persona_note = ""
    if slot.target_persona == "고객":
        persona_note = """
   **이 질문은 고객 본인이 자기 얘기를 하는 것이다 — 문장 끝까지 "제가/저는"을
   유지하고, 뒤에서 자기 얘기를 3인칭 "고객이/고객은"으로 바꿔 부르지 마라.**
   - 나쁜 예(뒷부분에서 3인칭으로 바뀜): "제가 계좌이동을 신청하려고 하는데,
     고객이 설정한 자동송금으로 증권 계좌에 주기적으로 송금되는 건
     계좌이동서비스로 옮길 수 있는 건가요?"
   - 좋은 예(끝까지 1인칭 유지): "제가 계좌이동을 신청하려고 하는데, 제가
     설정한 증권 계좌 자동송금도 포함해서 이동되나요?\""""
    elif slot.target_persona == "내부직원":
        persona_note = """
   **이 질문은 은행 내부 직원이 업무 처리 중 묻는 것이다.** 고객처럼
   "제가/저는"으로 시작하지 말고, 심사·취급·처리 절차를 확인하는 직원 관점
   ("~취급 시", "~처리 시", "~할 때", "~심사할 때")으로 물어라.
   - 나쁜 예(고객 어투로 새어나감): "제가 대출 심사를 받으려는데 담보인정비율이
     어떻게 적용되나요?"
   - 좋은 예(직원 관점 유지): "담보대출 취급 시 담보인정비율은 어떤 기준으로
     적용하나요?\""""

    term_avoidance_instruction = """
13. **발췌문에 등장하는 전문용어·법률용어를 그대로 쓰지 말고, 사용자가 실제로 쓸 법한
   일상어로 풀어써라** (상품·서비스 이름 자체는 규칙 5에 따라 그대로 언급해도 된다 —
   이 규칙은 그 상품·조건을 설명하는 전문용어·법률용어에만 적용된다). 모든 슬롯에
   예외 없이 적용되는 원칙이다.
   **질문 문장 안의 전문용어·법률용어는 하나도 빠짐없이 전부 풀어써야 한다 — 여러
   개 중 일부만 일상어로 바꾸고 나머지는 그대로 남겨두는 것도 규칙 위반이다.**
   - 나쁜 예(전문용어 그대로): "거래외국환은행 지정등록을 했는데 거래계좌 조건이 있나요?"
   - 나쁜 예(일부만 풀어씀 — "지정등록"은 남음): "해외송금 은행 지정등록을 했는데
     계좌 조건이 있나요?"
   - 좋은 예(전부 일상어로 풀어씀): "해외송금 등록을 마쳤는데 계좌 조건이 있나요?"
   **상품명처럼 고유명사로 착각하기 쉬운 은행 내부 식별자·행정 용어("실명번호",
   "실명확인번호", "고유식별번호" 등)도 전문용어다 — 상품·서비스 이름이 아니라면
   예외 없이 이 규칙 대상이다.**
   - 나쁜 예("실명번호"가 뭔지 모르는 사람은 이해 못 함): "제가 14세인데 실명번호가
     없으면 휴대폰 번호만으로 바이오인증 가입 절차가 어떻게 되나요"
   - 좋은 예(내부 용어 없이 상황만 말함): "제가 14세인데 바이오인증을 이용하고
     싶어요\""""

    doc_ids = {c.doc_id for c in slot.context()}
    multihop_boost = slot.difficulty == "상" and len(slot.context()) >= 3
    multi_doc_instruction = ""
    if len(doc_ids) > 1:
        multi_doc_instruction = f"""
12. **아래 발췌문은 서로 다른 문서 {len(doc_ids)}개에서 왔다. 이 문서들 내용을 각각
   다 활용해야만 답할 수 있는 질문으로 만들어라** (예: 두 상품 조건 비교, 한 절차의
   앞·뒤 단계가 문서별로 나뉜 경우). 한 문서만으로 답이 끝나는 질문은 안 된다.
   **가짜 통합 금지**: "두 번째 문서에는 예외 규정이 없다"처럼 답이 안 되는 문서를
   "확인해봤지만 없었다"는 문장으로만 끼워넣지 마라 — 형식만 맞춘 것이다. 두 문서
   각각이 **실질적으로 새로운 사실**을 제공해야 하고, 한쪽이라도 "해당 없음" 확인용
   이면 이 소재로는 질문을 만들 수 없다."""
        if multihop_boost:
            multi_doc_instruction += """
   **난이도 "상"이므로 가능하면 근거 3개 이상(조건/예외/제한사항 등)을 전부
   연결해야만 답이 나오는 질문을 우선 고려하라** — 근거 2개만 결합해도 답이
   나오는 얕은 결합보다, 3개 이상을 순차적으로 연결해야 하는 질문을 우선한다."""
    elif slot.difficulty in ("중", "상") and len(slot.context()) > 1:
        multi_doc_instruction = f"""
12. **아래 발췌문은 한 문서 안에서도 서로 다른 조항/주제 {len(slot.context())}개를
   담고 있다. 여러 조항 내용을 전부 결합해야만 답할 수 있는 질문으로 만들어라**
   (나머지 조항이 필요 없어지는 질문은 안 된다). 여러 조건을 하나의 자연스러운
   상황 안에서 같이 묻는 형태를 우선 고려한다.
   **가짜 통합 금지**: 한쪽 조항을 "해당 없음" 확인용으로만 끼워넣지 마라 — 각
   조항이 실질적으로 새로운 사실을 제공해야 한다."""
        if multihop_boost:
            multi_doc_instruction += """
   **난이도 "상"이므로 가능하면 근거 3개 이상을 전부 연결해야만 답이 나오는
   질문을 우선 고려하라.**"""

    intro = (
        f'아래 규칙을 지켜, 실제 고객·직원이 창구나 콜센터에서 말로 물어보는 듯한 자연스러운 발화 '
        f'서로 다른 후보 {batch_size}개를 만들어라.' if batch_size > 1 else
        "아래 규칙을 지켜, 실제 고객·직원이 창구나 콜센터에서 말로 물어보는 듯한 자연스러운 발화 1개를 만들어라."
    )
    output_instruction = (
        f'서로 다른 표현·각도의 질문 {batch_size}개를 JSON으로만 출력하라: '
        f'{{"candidates": ["...", "...", ...]}} (배열 길이 정확히 {batch_size}). '
        f'각 후보는 전부 위 규칙을 독립적으로 지켜야 한다. 설명이나 다른 텍스트는 출력하지 않는다.'
        if batch_size > 1 else
        "질문 문장만 출력하라. 설명, 따옴표, 번호매김 없이 한 줄만 출력한다."
    )

    return f"""당신은 신한은행 RAG 벤치마크용 질문을 만드는 어시스턴트다.
{intro} 절대 문서를 "읽는" 사람의 말투가 아니라 "궁금해서 물어보는"
사람의 말투여야 한다.

[필수 조건]
1. 발화체 1문장·1질문만 담는다. "~인지, ~인지, ~인지"처럼 의문형 어미를 나열해
   사실상 여러 개를 묻는 것도 금지 — 물음표가 1개여도 여러 질문이 섞이면 위반이다.
   "그리고/또한/아니면"처럼 연결어로 서로 다른 요청·항목 2개 이상을 한 문장에 묶어
   담는 것도 금지다 — 예를 들어 "계좌 항목, 담당자란 정보, 그리고 법적 근거를
   어떻게 기재하나요?"처럼 물음표가 1개여도 실질적으로 세 가지를 한꺼번에 묻는
   것은 위반이다.
2. 실제 사람이 창구·전화·챗봇에서 말할 법한 구어체로 쓴다. 물음표 없이 끝나도 된다
   (예: "해지 방법은?", "수수료는요?", "~인지 궁금해요", "~좀 알려주세요"). 어미가
   의문형("-나요/-되나요/-가요/-까요"나 "-요"만 뺀 "-나/-되나")이면 물음표가 없어도
   질문으로 들린다 — 아래 세 형태를 실제로 섞어 써라:
   - 순수 의문형: "~되나요?", "~인가요?"
   - 모른다는 진술형(평서형 어미): "~인지 잘 모르겠어요", "~인지 헷갈리네요"
   - 요청형: "~좀 알려주세요", "~인지 설명 좀 부탁드려요"
   존댓말이 아닌 경우 "-요"체만 반복하지 말고 편한 어미도 섞어라(예: "궁금하네", "적용되나",
   "알고 싶은데").
   **약관·규정 문서의 사무적 어미("~로 판단되는지", "~인정되는지", "~반영되나요",
   "~조치가 이루어지는지")를 그대로 옮기지 마라 — 실제 사람은 서류를 읽듯 말하지
   않는다.** 쉬운 동사·어미로 바꿔 써라.
   - 나쁜 예(문서 어미 그대로): "계약기간을 1년으로 약정할 수 없는지 잘 모르겠어요"
   - 좋은 예(쉬운 말로 바꿈): "1년만 납입하고 싶은데 가능한가요?"
{yes_no_note}
3. 개인정보(실명, 계좌번호, 주민번호, 전화번호처럼 실제 신원·거래를 특정할 수 있는
   정보)를 포함하지 않는다. **"제가 파산해서", "제가 창구직원인데", "회사용으로
   추가 가입하려는데"처럼 신원을 특정하지 않는 상황·역할·계기는 개인정보가 아니라
   규칙 9가 요구하는 정상적인 상황 설명이다 — 이런 것까지 개인정보로 오인해서
   빼지 마라.**
4. "종합적으로 설명해달라"류 포괄적 요구는 금지 — 짧고 목적이 분명한 구체적 사실을
   겨냥한다. **발췌문에 명시된 사실까지만 묻는다** — "왜/어떻게/무슨 기준으로"는
   그 메커니즘 자체가 발췌문에 적혀 있을 때만 묻는다(예: "A가 필요하다"까지만
   있으면 "어떻게 하나요?"라고 캐묻지 않는다 — 발췌문만으로 답할 수 없어진다).
   **여러 개를 서로 다르게 만들어야 한다고 해서, 처리기간·연락처·제출 부수·계산법·
   보관기간처럼 신청서·설명서에 흔히 있을 법하지만 이 발췌문엔 없는 항목을 지어내
   개수를 채우지 마라.** 발췌문이 실제로 뒷받침하는 사실이 적으면 후보 개수가
   적어도 된다 — 다양성보다 "발췌문에 실제로 있는가"가 항상 우선이다.
5. 지시어("이 문서", "이 설명서")로만 퉁치지 말고 실제 상품·주제를 언급하되,
   **문서 제목을 괄호까지 포함해 그대로 읽지 말고 입에서 나올 법하게 풀어서 말한다**
   (예: "CD수익률 설명서(예금거래용)" → "예금거래용 CD수익률", "가계대출 상품설명서
   (은행대리업용)" → "가계대출").
   **제목을 그대로 안 베끼는 것과 상품·주제 자체를 생략하는 것은 다르다 — 발췌문 없이
   질문만 읽는 사람도 어떤 상품·서비스·제도에 대한 질문인지 알 수 있어야 한다.**
   "예탁금", "신청서", "수수료"처럼 여러 상품에 공통될 수 있는 일반명사만 남기고
   상품·서비스명을 빼지 마라.
   - 나쁜 예(상품명 없이 일반명사만 남음): "신청서에 서명 대신 날인만 해도 되나요?"
     (어떤 신청서인지 안 나옴)
   - 좋은 예(상품명을 자연스럽게 포함): "계좌 개설 신청서에 서명 대신 날인만 해도
     되나요?"

   **짧은검색형처럼 문장을 줄일 때도 어미·수식어만 걷어내야지, 상품명이나 핵심
   주체를 지우면 안 된다.**
   - 나쁜 예(짧게 줄이려다 상품명이 통째로 사라짐): "우선순위 지급대상은?"
   - 좋은 예(짧아도 상품명은 남김): "동반성장론 지급 우선순위는?"

   **"동의"·"신청"·"거부"처럼 주체나 대상이 여러 갈래일 수 있는 행위 동사는,
   서비스명만 붙여놓고 끝내지 말고 누가/무엇에 대한 동의·신청인지까지 밝혀라**
   — 관련자가 둘 이상(예: 친권자·자녀)일 수 있는 서비스면 누구의 동의인지,
   동의 대상 자체가 낯선 사내 용어(예: "안심차단")면 그게 무엇을 위한
   동의인지를 질문 안에 최소한으로라도 풀어줘야 한다. 서비스명이 있어도 이
   두 가지가 빠지면 실질적으로 self-contained가 아니다.
   - 나쁜 예(동의 주체 불명): "SOL패밀리 동의 거부하면 서비스 이용 자체가
     불가능한가요?" (친권자 동의인지 자녀 동의인지 안 나옴)
   - 좋은 예(주체를 명시): "자녀 계좌를 조회하려고 하는데, 제가 개인정보 제공에
     동의 안 하면 SOL패밀리 서비스를 못 쓰나요?"
   - 나쁜 예(동의 대상 불명 — "필요하다"를 "안 하면 안 되나요"로 바꾼 것뿐이라
     실질적 새 정보를 묻지도 않음): "제가 동의 안 하면 안심차단 조회가 안
     되나요?" (안심차단이 뭔지, 무엇을 조회한다는 건지 전혀 안 나옴)
   - 좋은 예(동의 대상을 풀어서 명시): "비대면으로 계좌를 개설하려는데, 안심차단
     등록 여부 조회에 동의 안 하면 계좌개설이 안 되나요?"

   **이 규칙은 "동의/신청/거부"에만 해당하지 않는다 — 질문 속 핵심 명사·동사가
   발췌문 안에서 여러 갈래를 가리킬 수 있다면(사고신고엔 분실/도난/지급거절
   이의제기 등, "이기다"엔 소송/분쟁조정 등 여러 상황이 있을 수 있다), 발췌문이
   실제로 다루는 그 갈래로 좁혀라 — 명사 하나로는 안 끝난다.**
   - 나쁜 예(무슨 신청·무슨 수수료인지 안 나옴): "비대면 신청 때 수수료 납부는?"
   - 나쁜 예(무슨 사고신고인지 안 좁혀짐): "사고신고된 전자채권 담보금은 은행이
     누구를 위해 보관하나요"
   - 나쁜 예("대상"이 무엇의 대상인지 없음): "제가 2023년 8월에 임대차 계약한
     집은 대상인지 잘 모르겠어요" (무슨 제도·심사의 대상인지가 빠짐)
   - 나쁜 예("결정일"이 무엇을 결정하는 날인지 없음): "제가 30일에 예금했는데
     해당일이 없으면 결정일이 어떻게 되는지 궁금해요" (이자 산정 기준일인지
     만기일인지 등 무슨 결정인지가 빠짐)
6. **조항·항·호 번호("제16조", "제3항", "2호" 등)를 절대 인용하지 않는다.** 실제
   고객·직원은 조항 번호를 보며 말하지 않는다 — 그 조항이 다루는 내용(주제)만으로
   자연스럽게 묻는다. **상품·포트폴리오·코스 이름 뒤에 붙는 순번("포트폴리오 3",
   "1형", "코스 2" 등)도 같은 문제다** — 그 번호가 무엇을 기준으로 매겨졌는지(위험도·
   구성비율 등)를 발췌문 밖 사람은 알 길이 없어, 조항 번호를 인용하는 것과 똑같이
   불친절하다. 번호만으로 묻지 말고 그 항목을 구별 짓는 실제 특징으로 묻거나,
   번호가 꼭 필요하면 그 특징을 함께 풀어써라.
   - 나쁜 예(번호만 인용, 무엇을 가리키는지 알 수 없음): "적극투자형 포트폴리오 3
     유효기간은?"
   - 좋은 예(번호 대신/함께 특징으로 특정): "적극투자형 포트폴리오 중 주식 비중이
     가장 높은 유형의 유효기간은?"
7. **발췌문의 구체적 답 내용(수치·기간·조건·산식·조건+결과·결론·여러 조항의
   나열 등)을 질문 문장 안에 이미 적어놓고 "~인가요?/~맞나요?"로 확인만 요구하는
   폐쇄형을 만들지 마라** — 답에 해당하는 부분은 통째로 빼고 "무엇을/어떻게/얼마나"
   로 여는 개방형으로 만들거나, 마지막 조건 하나는 열어둬서 답하는 쪽이 채우게
   하라. 산식은 세부 수치 없이 "계산 방식이 같은가"로, 서로 다른 조항은 나열해
   옮기지 말고 그 결합이 만드는 구체적 상황을 직접 물어라. **"~인가요?/~되나요?"
   자체는 문제가 아니다**("면제되나요?"는 정상) — 문제는 답 내용이 질문에 이미
   있는 경우뿐이다.
   - 나쁜 예(조건+결과를 통째로 채워넣고 확인만 요구): "장외파생상품 투자정보
     확인서에 거래한 상품 종류랑 거래 규모를 적으면 은행이 그 정보를 근거로
     투자등급을 산출하고 확인서 사본을 주는 건가요?"
   - 좋은 예(조건까지만 묻고 결과는 열어둠): "장외파생상품 확인서 작성할 때
     뭘 적어?"
   - 나쁜 예(결론을 진술문으로 미리 박아넣음): "가입대상이 제한없다고 되어
     있는데, 그러면 이 상품은 비과세 혜택을 받을 수 없는 건가요?"
   - 좋은 예(결론 없이 순수하게 열어서 물음): "가입대상이 제한없다고 되어 있는
     상품인데, 비과세 혜택도 받을 수 있나요?"
   **"~라는데"처럼 전해 들은 정보를 서두에 깔아놓는 방식도 결론 선진술과 같다** —
   전해 들었다는 화법으로 감싸도, 그 안에 담긴 내용이 곧 답이면 여전히 폐쇄형이다.
   - 나쁜 예("-라는데"에 답의 핵심(원금 손실 가능)을 이미 담아놓고 확인만 요구):
     "적립식 신탁은 투자 성과 때문에 수익이 달라지고 원금 손실도 날 수 있다는데,
     제가 맡긴 원금은 은행 예금처럼 보장되나요?"
   - 좋은 예(전해들은 정보 없이 순수하게 열어서 물음): "적립식 신탁을 이용하면
     투자 성과에 따라 원금이 바뀌나요? 아님 원금은 보장되나요?"
8. **가운뎃점(·)이나 괄호로 부가설명을 덧붙이는 문어체 기호를 쓰지 마라**
   ("환율·수수료" → "환율이나 수수료", "송금(현금수취)" → 괄호 없이 풀어 쓰거나
   생략).
   **하나의 질문 안에 서로 다른 여러 원인·조건을 "~하거나 ~하면"처럼 나열해 엮지
   마라** — 딱 하나의 상황만 가정하고 하나의 결과만 물어라. 여러 사유를 나열한
   뒤 "그리고 책임은 어떻게 되나요"처럼 별개 질문을 덧붙이는 것도 금지한다.
   - 나쁜 예(복수 사유 나열 + 질문도 2개): "제가 모임을 운영 중인데, 모임장이
     특정 모임원을 내보내거나 모임통장이 압류·거래중지·휴면 등으로 서비스가
     중단되면 그 중단 사유가 해소되었을 때 서비스는 어떻게 재개되고 모임장과
     은행의 책임은 어떻게 되는 건가요?"
   - 좋은 예(사유 하나만 가정, 질문도 하나): "제가 모임을 운영 중인데, 모임장이
     특정 모임원을 내보내서 서비스가 중단되면 다시 재개할 수 있나요?"
   - 나쁜 예(괄호 부가설명 + 복수 조건): "제가 신한 앱으로 캄보디아에 전화번호
     송금(현금수취)하려는데 제 출금계좌가 압류되거나 잔액이 부족하면 현지 Wing
     에이전트가 영업시간에 있어도 현금 지급이 제한되나요?"
   - 좋은 예(괄호 없이, 조건도 하나만): "제가 신한 앱으로 캄보디아에 전화번호
     송금하려는데 제 출금계좌가 잔액이 부족하면 현금 지급이 제한되나요?"
9. 상황 설명("제가 ~하려는데" 등)을 붙일 때는 상투적 도입부로 끝내지 마라 —
   **왜 이 상황에 놓였는지(계기·이유)나 무엇을 시도했다가 어떤 결과·문제를
   겪었는지(시도한 방법 → 부딪힌 벽)** 중 최소 하나를 담아야 뒤에 오는 질문이
   자연스럽게 이어진다. 신원을 특정 안 하는 계기·역할(파산, 이직, 창구직원 등)은
   규칙 3의 개인정보가 아니다. (짧은검색형·확인형처럼 상황 없이 다짜고짜 묻는
   스타일도 규칙 15의 다양성을 위해 섞되, 상황을 붙이기로 했으면 위 요건을
   채워라.) 상황을 구체화하면 원래 겨냥하던 사실과 질문이 정확히 같지 않아도
   된다 — 그 상황에서 자연스럽게 다음으로 궁금해할, 발췌문 안의 인접한 사실로
   옮겨가도 된다.
   **상황(계기)은 나이·역할·이미 한 행동처럼 화자가 실제로 아는 자기 얘기여야 한다
   — 발췌문의 자격 조건("~가 없으면", "~에 해당하면")을 화자가 스스로 체크한 것처럼
   질문 서두에 그대로 옮기지 마라.** 그 조건을 충족하는지 여부 자체가 발췌문
   지식이 있어야 판단 가능한 것이라, 실제 고객은 그런 전제를 달고 묻지 않는다 —
   조건은 답변이 알려줄 몫이지 질문이 미리 자가진단할 몫이 아니다.
   - 나쁜 예(문서의 자격 조건을 화자가 스스로 판정): "제가 14세인데 실명번호가
     없으면 휴대폰 번호만으로 바이오인증 가입 절차가 어떻게 되나요"
   - 좋은 예(자기 사실만 말하고 조건 판단은 답변에 맡김): "제가 14세인데
     바이오인증을 이용하고 싶어요"
   - 나쁜 예(계기·시도 없이 사실만 물음): "예금토큰에서 서면 신고로만 변경해야
     하는 항목들이 뭐가 있나요"
   - 좋은 예(계기+시도한 방법 포함, 질문도 자연스러운 다음 궁금증으로 이동):
     "제가 파산해서 예금토큰 변경하려는데 앱으로만 하려니까 서면신고하라고
     하네요. 제가 서면신고를 위해 뭘 준비해야하나요?"
   - 나쁜 예(계기 없이 가능여부만 확인): "제가 개인명과 회사명으로 국민주택채권
     포털 ID를 각각 만들 수 있나요?"
   - 좋은 예(기존 상태 + 새로 필요해진 이유): "개인으로 국민주택채권이 있는데
     회사에서 필요한거라 추가로 회원가입 가능해?"
   - 나쁜 예(역할·상황 없이 일반 절차만 확인): "의심거래보고가 접수되면
     금융정보분석원이 이메일로 통보하나요?"
   - 좋은 예(역할+상황 후 인접 사실로 질문이 이동): "제가 창구직원인데
     의심거래보고가 접수되었어요. 혹시 금융정보분석원에 직접 가서 관련 서류를
     모두 제출해야 하나요?"

   **조건을 앞에 몰아넣고 끝에 물음표 하나로 마무리하지 마라** — 핵심 질문을
   먼저 정하고 상황·조건은 문장 중간에 섞어라. 같은 이유로, 은행이 정하는 결과
   (거절 사유 등)를 화자가 이미 안다는 듯 완료형("~된")으로 기정사실화하지 말고
   "~하면"류 가정형으로 열어라.
   - 나쁜 예(조건 몰아넣기): "제가 대출을 연체해서 보증보험사가 대신 갚아주고
     그 권리가 보증보험사로 넘어갔는데, 은행이 갖고 있던 채권이랑 근저당권까지
     전부 보험사한테 넘어가는 거고, 나중에 보험사가 그 돈을 회수하고 남는 게
     있으면 저한테 돌려주는 건가요"
   - 좋은 예: "보증보험사가 제 대출을 대신 갚아주면 은행 채권도 넘어간다는데,
     나중에 회수하고 남은 돈은 돌려받을 수 있나요"
   - 나쁜 예(완료형 기정사실화): "분쟁 때문에 지급이 거절된 전자채권은 신고서에
     뭐 적어야 하나요"
   - 좋은 예(가정형으로 열어서 물음): "전자채권이 상대방과 합의가 안 돼서 지급이
     거절되면 신고서에 뭐라고 적어야 하나요"{persona_note}
10. {data_type_instruction}
11. **발췌문 문장 하나를 그대로 옮겨 묻지 말고, 실제로 추론이 필요하도록 만들어라**
   — 아래 중 하나는 반드시 요구해야 한다.
   (a) 서로 다른 사실 두 개를 결합해야 나오는 결론(결합 결과 자체가 발췌문에
   그대로 적혀 있지 않아야 함),
   (b) 발췌문의 일반 조건·기준을 구체적 상황에 대입했을 때의 결과 판단,
   (c) 여러 값·기준을 비교해서 나오는 차이·우열 판단.
   두 사실을 나란히 묻기만 하면(예: "A는 얼마고 B는 뭔가요?") 진짜 추론 질문이
   아니다 — 하나의 결론으로 수렴하도록 만들어라.
   - 나쁜 예(사실 나열, 결합 불필요): "중도상환수수료율이 얼마고 면제 조건은
     뭔가요?"
   - 좋은 예(상황 대입 추론 필요): "3년 전에 대출받았는데 지금 갚으면 중도상환
     수수료를 내야 하나요?"

   **발췌문에 일수·나이·금액·기간처럼 구체적 수치가 조건으로 나오면, (b) 상황
   대입은 그 수치를 그대로 옮겨 적는 게 아니라 다른 값으로 바꿔서 물어야 한다**
   — 발췌문 수치를 그대로 쓰고 "~되나요?"로만 물으면 그냥 그 문장을 확인하는
   것일 뿐 대입이 아니다. 실제 판정 대상은 발췌문에 없는 값이어야 진짜 추론이 된다.
   - 나쁜 예(발췌문 수치 "14일"을 그대로 씀): "대출 취급 시 이자 14일 연체만으로
     모든 채무의 기한상실이 되나요?"
   - 좋은 예(다른 일수로 대입): "대출에서 12일 연체되면 어떻게 되나요?"
   - 나쁜 예(발췌문 나이 "65세"를 그대로 씀, 상품명도 생략): "65세 전화계약
     청약철회는?"
   - 좋은 예(다른 나이로 대입 + 상품명 포함): "70세인데 미래에셋생명 전화로
     청약철회도 가능해?"
{term_avoidance_instruction}
15. **질문 스타일도 아래 6가지 중에서 골고루 섞어서 써라** (매번 같은 형태로 몰리지 않게):
{_QUERY_STYLE_GUIDE}
   짧은검색형·불완전형을 쓸 때는 상황 설명 서두("제가 ~하려는데")나 존댓말 요청형
   어미("~해주실래요", "~부탁드려요")를 붙이지 말고 명사구+반말 어미로 바로 끝내라
   (예: "전세대출 중도상환수수료는?"). 내부 약어·코드성 표현(PG 등)보다
   일상어(결제대행 등)를 우선 써라.
   **짧다고 해서 "[상품/서비스명] 목적은?/대상은?/누구?"처럼 발췌문의 정의·제목을
   그대로 되묻기만 하는 빈 질문을 만들면 안 된다** — 발췌문이 실제로 담고 있는
   구체적 조건·기준·수치·절차 중 하나를 짧게 겨냥해야 한다.
   - 나쁜 예(정의를 그대로 되묻기만 함, 실질 정보 없음): "노란우산공제 가입 대상은?"
   - 좋은 예(구체적 항목을 겨냥): "노란우산공제 중도해지 위약금은?"
{multi_doc_instruction}

[금지 예시]
{chr(10).join(f"- {e}" for e in _BAD_EXAMPLES)}

[좋은 예시]
{chr(10).join(f"- {e}" for e in _GOOD_EXAMPLES)}

{output_instruction}"""


def build_user_prompt(slot: QuerySlot, focus_topic: str, feedback: Optional[str] = None) -> str:
    # 문서명을 라벨로 그대로 보여주면 모델이 그 표기를 그대로 베끼는 경향이 있어,
    # 참고용으로만 아래에 옮겨적고 "그대로 인용하지 말라"는 지시를 함께 준다.
    doc_names = ", ".join(dict.fromkeys(c.doc_nm for c in slot.context()))
    excerpt = "\n\n".join(c.content for c in slot.context())
    prompt = (
        f"다음 발췌문(출처 문서: {doc_names} — 이 제목을 그대로 인용하지 말고 자연스럽게 풀어 말할 것)을 "
        f"근거로, '{focus_topic}'에 대한 질문을 1개 만들어라.\n\n[발췌문]\n{excerpt}\n"
    )
    if feedback:
        prompt += f"\n[이전 시도 실패 사유 — 반드시 개선할 것]\n{feedback}\n"
    return prompt


# Q4 — 규칙 기반 1차 필터
_PII_PATTERNS = [
    re.compile(r"\d{2,4}-\d{3,4}-\d{4}"),
    re.compile(r"\d{6}-\d{7}"),
    re.compile(r"\d{2,6}-\d{2,6}-\d{2,8}"),
]
_FORBIDDEN_PHRASES = ["종합적으로", "전반적으로", "관련 규정 전체", "전부 설명", "다 알려주"]
_LISTING_CONNECTORS = ["그리고 또", "또한 ", "그리고 ", "아울러 ", "덧붙여"]
# "~인지/~는지"는 간접의문문 어미. 한 문장에 2번 이상 나오면 사실상 여러 질문을
# 나열한 것으로 본다 (예: "용도는 무엇인지, 정의는 무엇인지, 발생일은 언제인지").
_INDIRECT_Q_RE = re.compile(r"(?:인지|는지)")
# "이 문서/이 설명서/이 약관"처럼 지시어로만 대상을 가리키거나("이" 유무 무관 —
# "문서에 보면"처럼 지시어 없이 그냥 "문서"라고만 불러도 실제 명칭이 없기는 매한가지),
# 실제 상품·문서명을 대지 않고 일반명사만 쓰는 패턴.
_VAGUE_REF_RE = re.compile(
    r"(?:이\s*)?(?:문서|설명서|약관|자료|안내문|서식|신청서|계약서|안내서|확인서|명세서)(?:가|는|를|에|의)?"
)
# 조항·항·호 번호를 그대로 인용하는 패턴 (예: "제16조", "제3항", "2호").
_LEGAL_CITATION_RE = re.compile(r"제\s*\d+\s*(?:조|항|호)")
# 가운뎃점(·)은 프롬프트로만 금지했더니 다시 새어 나온 문어체 나열 기호라, 규칙
# 필터로도 확실히 걸러낸다 (예: "압류·거래중지·휴면").
_MIDDLE_DOT_RE = re.compile(r"[·・]")
# 괄호로 부가 설명을 덧붙이는 것도("송금(현금수취)", "CD수익률 설명서(예금거래용)"처럼)
# 구어체가 아니므로 용도를 가리지 않고 전부 거른다.
_PAREN_RE = re.compile(r"[()（）]")
# 서술어·의문 어미 없이 명사(구)만 나열하고 끝나는 "제목형" 문구를 거른다
# (예: "사고신고 담보금 예치증 내용", "지급정지 요청서 필드 목록 확인" — 실제 발화가
# 아니라 문서 제목·메모처럼 읽힘). "?"로 끝나거나, 흔한 한국어 술어 종결 음절로
# 끝나면 정상 발화로 보고 통과시킨다("해지 방법은?", "수수료는요?"처럼 명사(구) +
# 물음표는 정상이므로 "?" 유무를 우선 확인한다).
_PREDICATE_ENDING_RE = re.compile(r"[요다까나가지네죠게래라데냐니음줘야좀봐든고어려]$")


def rule_based_filter(query: str) -> tuple[bool, Optional[str]]:
    # 물음표는 필수가 아니다 — "~인지 궁금해요"처럼 물음표 없는 요청·진술형도 정상.
    # 물음표가 2개 이상이면 질문이 여러 개 섞였다는 신호이므로 그 경우만 거른다.
    if query.count("?") >= 2:
        return False, "질문 부호(?)가 2개 이상 — 질문이 여러 개 섞인 것으로 의심됨"
    for pattern in _PII_PATTERNS:
        if pattern.search(query):
            return False, "개인정보로 의심되는 숫자 패턴 포함"
    if _MIDDLE_DOT_RE.search(query):
        return False, "가운뎃점(·) 등 문어체 나열 기호 포함"
    if _PAREN_RE.search(query):
        return False, "괄호로 부가 설명을 덧붙임 - 구어체가 아님"
    for phrase in _FORBIDDEN_PHRASES:
        if phrase in query:
            return False, f"포괄적 요구 표현 '{phrase}' 포함"
    if any(c in query for c in _LISTING_CONNECTORS) and any(c in query for c in ["도 궁금", "각각"]):
        return False, "나열형 복합 질문으로 의심됨"
    if len(_INDIRECT_Q_RE.findall(query)) >= 2:
        return False, "'~인지/~는지'가 2회 이상 나와 여러 질문을 나열한 것으로 의심됨"
    if _VAGUE_REF_RE.search(query) and not re.search(
        r"[가-힣A-Za-z0-9]{2,}\s?(설명서|약관|안내|규정|서비스|상품|계좌|예금|대출|서식|신청서|계약서|확인서|명세서)", query
    ):
        return False, "'이 문서/이 서식' 등 지시어만 있고 실제 상품명·문서명이 없음"
    if _LEGAL_CITATION_RE.search(query):
        return False, "조항·항·호 번호(제N조 등)를 그대로 인용함 - 실제 발화라면 조항 번호를 말하지 않음"
    if len(query.strip()) < 8:
        return False, "질문이 지나치게 짧음"
    stripped = query.strip()
    if not stripped.endswith("?") and not _PREDICATE_ENDING_RE.search(stripped):
        return False, "서술어·의문 어미 없이 명사(구)로만 끝나는 제목형 문구 - 실제 발화 형태가 아님"
    return True, None


# Q5 — LLM-judge 검증
def llm_judge(llm: LLMClient, query: str, slot: QuerySlot) -> tuple[bool, Optional[str]]:
    excerpt = "\n\n".join(c.content for c in slot.context())
    multihop_boost = slot.difficulty == "상" and len(slot.context()) >= 3
    system = (
        _JUDGE_SYSTEM
        + (_REASONING_ADDENDUM if slot.data_type == "pos_neg" else "")
        + _TERM_AVOIDANCE_ADDENDUM
        + (_MULTIHOP_ADDENDUM if multihop_boost else "")
    )
    result = llm.complete_json(system, f"[발췌문]\n{excerpt}\n\n[질문]\n{query}", reasoning_effort="medium")
    if (
        result.get("answerable_from_context")
        and result.get("single_natural_intent")
        and result.get("self_contained")
        and result.get("open_ended")
        and result.get("requires_reasoning", True)
        and result.get("terms_paraphrased")
        and result.get("multi_hop_required", True)
    ):
        return True, None
    return False, result.get("reason", "LLM judge 판정 실패")


def batch_llm_judge(llm: LLMClient, queries: list[str], slot: QuerySlot) -> list[tuple[bool, Optional[str]]]:
    """queries 전부를 API 호출 1번으로 판정한다 (Q3에서 여러 후보를 한 번에 만들었을 때,
    검증도 한 번에 끝내 왕복 횟수를 줄인다). 반환 리스트는 queries와 순서가 같다."""
    excerpt = "\n\n".join(c.content for c in slot.context())
    numbered = "\n".join(f"{i+1}. {q}" for i, q in enumerate(queries))
    multihop_boost = slot.difficulty == "상" and len(slot.context()) >= 3
    system = (
        _BATCH_JUDGE_SYSTEM
        + (_REASONING_ADDENDUM if slot.data_type == "pos_neg" else "")
        + _TERM_AVOIDANCE_ADDENDUM
        + (_MULTIHOP_ADDENDUM if multihop_boost else "")
    )
    result = llm.complete_json(system, f"[발췌문]\n{excerpt}\n\n[질문 목록]\n{numbered}", reasoning_effort="medium")
    results = result.get("results", [])

    verdicts: list[tuple[bool, Optional[str]]] = []
    for i in range(len(queries)):
        r = results[i] if i < len(results) else {}
        ok = (
            r.get("answerable_from_context")
            and r.get("single_natural_intent")
            and r.get("self_contained")
            and r.get("open_ended")
            and r.get("requires_reasoning", True)
            and r.get("terms_paraphrased")
            and r.get("multi_hop_required", True)
        )
        verdicts.append((bool(ok), r.get("reason", "LLM judge 판정 실패" if not ok else None)))
    return verdicts


# Q7 — persona 사후 분류
_VALID_QUERY_STYLES = ("정보형", "확인형", "짧은검색형", "문제해결형", "절차문의형", "불완전형")


def classify_query_attributes(llm: LLMClient, query: str) -> tuple[Persona, QueryStyle]:
    result = llm.complete_json(_QUERY_ATTRS_SYSTEM, query, reasoning_effort="low")
    persona = result.get("persona")
    persona = persona if persona in ("고객", "내부직원") else "고객"
    style = result.get("query_style")
    style = style if style in _VALID_QUERY_STYLES else "정보형"
    return persona, style


def batch_classify_query_attributes(llm: LLMClient, queries: list[str]) -> list[tuple[Persona, QueryStyle]]:
    """queries 전부를 API 호출 1번으로 persona/style 분류한다 (classify_query_attributes의
    배치 버전, batch_llm_judge와 같은 패턴). target_persona가 지정된 슬롯에서 후보마다
    순차로 호출하면 목표 persona가 뒤쪽 후보에서만 나올 때 batch_size만큼 순차 API 왕복이
    생겼는데, 그걸 호출 1번으로 줄인다. 반환 리스트는 queries와 순서가 같다."""
    numbered = "\n".join(f"{i+1}. {q}" for i, q in enumerate(queries))
    result = llm.complete_json(_BATCH_QUERY_ATTRS_SYSTEM, f"[질문 목록]\n{numbered}", reasoning_effort="low")
    results = result.get("results", [])

    attrs: list[tuple[Persona, QueryStyle]] = []
    for i in range(len(queries)):
        r = results[i] if i < len(results) else {}
        persona = r.get("persona")
        persona = persona if persona in ("고객", "내부직원") else "고객"
        style = r.get("query_style")
        style = style if style in _VALID_QUERY_STYLES else "정보형"
        attrs.append((persona, style))
    return attrs


# 오케스트레이터
class QueryGenerationFailed(Exception):
    pass


class QueryGenerator:
    def __init__(self, llm: LLMClient, cfg: Optional[QueryGenConfig] = None):
        self.llm = llm
        self.cfg = cfg or QueryGenConfig()

    def generate(self, slot: QuerySlot) -> QueryResult:
        focus_topic = select_focus_topic(slot)
        batch_size = self.cfg.batch_size
        system_prompt = build_system_prompt(slot, batch_size=batch_size)
        # 나머지 랜덤 사용처(main.py의 item_seed/pool_seed 등)와 같은 방식으로 슬롯별
        # 시드를 고정한다 — 같은 슬롯을 다시 돌려도 후보 채택 순서가 재현되게 함.
        rng = random.Random(hash(slot.index) % (2**31))

        feedback: Optional[str] = None
        last_fail_reason: Optional[str] = None
        fallback: Optional[tuple[str, Persona, QueryStyle]] = None

        for attempt in range(1, self.cfg.max_attempts + 1):
            user_prompt = build_user_prompt(slot, focus_topic, feedback)

            batch_error: Exception | None = None
            if batch_size <= 1:
                with timed("Query 생성"):
                    draft = self.llm.complete(system_prompt, user_prompt, temperature=self.cfg.temperature).strip()
                candidates = [draft]
            else:
                try:
                    # max_tokens는 설정된 경우에만 넘긴다 — 기본값(None)일 때는 이전과 완전히
                    # 동일한 호출이 되게 해서(LLMClient 구현체가 max_tokens 파라미터를 모르는
                    # 경우와도 호환), 이 override를 명시적으로 켠 경우에만 영향이 생기게 한다.
                    extra = {"max_tokens": self.cfg.max_tokens} if self.cfg.max_tokens is not None else {}
                    with timed("Query 생성"):
                        result = self.llm.complete_json(system_prompt, user_prompt, **extra)
                    candidates = [c.strip() for c in result.get("candidates", []) if c and c.strip()]
                except Exception as e:  # noqa: BLE001 - 배치 생성 실패도 재시도 대상으로 흡수
                    logger.debug("[%s] attempt=%d 배치 생성 실패: %s", slot.index, attempt, e)
                    candidates = []
                    batch_error = e

            if not candidates:
                # batch_error가 있으면(API 호출/JSON 파싱 자체가 실패) 원인을 그대로 남긴다 —
                # 이게 없으면 "왜" 실패했는지가 generic 메시지 뒤에 묻혀서, main.py의 build_item이
                # 남기는 최종 경고(QueryGenerationFailed의 str())만 봐서는 API 오류인지 단순히
                # 빈 배열 응답인지 구분이 안 됐다(pipeline.log에서 진단 불가능했던 지점).
                feedback = (
                    f"질문 후보 생성 API 호출 자체가 실패했습니다({batch_error}). 다시 생성해주세요."
                    if batch_error is not None else
                    "질문 후보가 생성되지 않았습니다(빈 배열 응답). 다시 생성해주세요."
                )
                last_fail_reason = feedback
                continue

            # Q4: 규칙 필터를 먼저 통과한 후보만 LLM judge로 넘긴다 (비용 절감)
            rule_passed: list[str] = []
            rule_fail_reasons: list[str] = []
            for c in candidates:
                ok, reason = rule_based_filter(c)
                if ok:
                    rule_passed.append(c)
                else:
                    rule_fail_reasons.append(f"{c!r}: {reason}")

            if not rule_passed:
                feedback = "모든 후보가 규칙 위반이었습니다 — " + "; ".join(rule_fail_reasons)
                last_fail_reason = feedback
                logger.debug("[%s] attempt=%d 배치 전체 규칙 필터 실패: %s", slot.index, attempt, feedback)
                continue

            # Q5/Q7: judge와 persona/어투 분류는 서로의 판정 결과에 의존하지 않는 독립 판정이라
            # ("규칙 통과 후보 리스트"만 읽고 각자 판단) 동시에 쏴서 왕복 1회를 없앤다.
            # ("생성" 콜 자체에 persona 판정까지 맡기지는 않는다 — 자기가 쓴 걸 자기가
            # 채점하는 꼴이라 안전하지 않다.) persona/style은 judge 통과 여부와 무관하게
            # rule_passed 전체를 분류해둔다 — judge에서 떨어질 후보까지 분류하는 약간의
            # 낭비는 있지만, 왕복 1회를 없애는 이득이 더 크다.
            def _run_judge():
                with timed("Judge"):
                    if len(rule_passed) == 1:
                        return [llm_judge(self.llm, rule_passed[0], slot)]
                    return batch_llm_judge(self.llm, rule_passed, slot)

            def _run_attrs():
                with timed("Persona"):
                    return batch_classify_query_attributes(self.llm, rule_passed)

            with ThreadPoolExecutor(max_workers=2) as pool:
                judge_future = pool.submit(_run_judge)
                attrs_future = pool.submit(_run_attrs)
                verdicts = judge_future.result()
                attrs = attrs_future.result()

            passing = [(c, attr) for c, (ok, _), attr in zip(rule_passed, verdicts, attrs) if ok]

            if not passing:
                judge_fail_reasons = [f"{c!r}: {r}" for c, (ok, r) in zip(rule_passed, verdicts) if not ok]
                feedback = "이번 배치 후보 전부 judge 거부됨 — " + "; ".join(judge_fail_reasons)
                last_fail_reason = feedback
                logger.debug("[%s] attempt=%d 배치 전체 judge 실패: %s", slot.index, attempt, feedback)
                continue

            # 통과한 후보가 여러 개면 항상 같은 순서(첫 번째)만 채택하지 않도록 섞는다 —
            # 배치 안에 실제로 다양한 어투가 섞여 있어도 선택 단계에서 모델의 "기본값"
            # 어투로 수렴하는 것을 막기 위함. persona/style을 이미 함께 들고 있으므로
            # (candidate, (persona, style)) 쌍째로 섞어 매핑이 깨지지 않게 한다.
            rng.shuffle(passing)

            if slot.target_persona is None:
                candidate, (persona, query_style) = passing[0]
                logger.debug(
                    "[%s] 질문 생성 성공 (attempt=%d, batch=%d개 중 채택, persona=%s, style=%s): %r",
                    slot.index, attempt, len(candidates), persona, query_style, candidate,
                )
                return QueryResult(
                    query=candidate, persona=persona, query_style=query_style,
                    focus_topic=focus_topic, attempts=attempt, passed=True,
                )

            # Q7: persona/style은 위에서 judge와 동시에 이미 분류해뒀으므로, 여기서는 그 결과
            # 중 목표 persona와 일치하는 첫 후보를 고르기만 한다(추가 LLM 호출 없음).
            for candidate, (persona, query_style) in passing:
                if persona == slot.target_persona:
                    logger.debug(
                        "[%s] 질문 생성 성공 (attempt=%d, batch=%d개 중 채택, persona=%s, style=%s): %r",
                        slot.index, attempt, len(candidates), persona, query_style, candidate,
                    )
                    return QueryResult(
                        query=candidate, persona=persona, query_style=query_style,
                        focus_topic=focus_topic, attempts=attempt, passed=True,
                    )
                if fallback is None:
                    fallback = (candidate, persona, query_style)

            # 이번 attempt의 후보 전부 목표 persona와 달랐다 — 재분류가 아니라 실제로
            # 다시 생성하도록 feedback을 남기고 다음 attempt로 넘어간다.
            feedback = (
                f"생성된 질문이 목표 관점({slot.target_persona})이 아니라 다른 관점으로 "
                f"읽혔습니다 — {slot.target_persona} 관점의 어투로 다시 써주세요."
            )
            last_fail_reason = feedback
            logger.debug("[%s] attempt=%d 목표 persona(%s) 불일치, 재시도", slot.index, attempt, slot.target_persona)

        if fallback is not None:
            candidate, persona, query_style = fallback
            logger.debug(
                "[%s] persona 목표(%s) 끝내 불일치 — 폴백 채택(persona=%s, style=%s): %r",
                slot.index, slot.target_persona, persona, query_style, candidate,
            )
            return QueryResult(
                query=candidate, persona=persona, query_style=query_style,
                focus_topic=focus_topic, attempts=self.cfg.max_attempts, passed=True,
            )

        raise QueryGenerationFailed(
            f"[{slot.index}] {self.cfg.max_attempts}회 재시도 후에도 질문 생성 실패: {last_fail_reason}"
        )


# 데모 (MockLLMClient)
if __name__ == "__main__":
    from sampling_plan import load_sampling_config, build_sampling_plan, resolve_positive_chunk_count

    # --- 실제 파이프라인 순서: config → 700행 슬롯 플랜 → 슬롯 하나 선택 ---
    cfg = load_sampling_config("config/sampling_plan.yaml")
    plan_df = build_sampling_plan(cfg)
    slot_row = plan_df[plan_df["data_type"] == "pos_neg"].iloc[0]

    n_positive = resolve_positive_chunk_count(slot_row["data_type"])
    # n_positive는 §4-1에서 실제 청크 풀(chunk_pool[category])을 대상으로 근거를 몇 개
    # 뽑을지 정하는 값이다. 여기서는 청크 풀이 없으므로 데모용 청크 1개로 대체한다.

    sample_chunk = ChunkRef(
        chunk_id="112db6af8a581762_5",
        doc_id="112db6af8a581762",
        doc_nm="가계대출 상품설명서 (은행대리업용)",
        category=slot_row["category"],
        subcategory="개인여신",
        content=(
            "중도상환해약금은 대출을 만기 전에 상환할 경우 부과되는 수수료로, "
            "대출 실행일로부터 3년 이내 상환 시에만 적용되며 3년 경과 후에는 면제된다. "
            "산출식은 (중도상환금액 x 중도상환해약금율 x 잔존일수/대출기간)이다."
        ),
        topic_list=["중도상환해약금 정의", "중도상환해약금 면제 조건", "중도상환해약금 산식"],
    )

    slot = QuerySlot(
        index=slot_row["index"],
        data_type=slot_row["data_type"],
        difficulty=slot_row["difficulty"],
        positive_chunks=[sample_chunk],   # 실제로는 n_positive개를 §4-1 로직으로 선정
        target_persona=slot_row["target_persona"],
    )

    mock = MockLLMClient(canned_queries=[
        "종합적으로 설명해 주세요",
        "대출 갚을 때 중도상환수수료는 언제까지 내야 하고 언제부터 안 내도 되나요?",
    ])

    generator = QueryGenerator(llm=mock, cfg=QueryGenConfig(max_attempts=3))
    result = generator.generate(slot)

    print(f"\n=== 슬롯(from sampling_plan.yaml): {dict(slot_row)} ===")
    print(f"positive_chunk_count 목표: {n_positive}")
    print("\n=== 최종 결과 ===")
    print(f"query   : {result.query}")
    print(f"persona : {result.persona}")
    print(f"focus   : {result.focus_topic}")
    print(f"attempts: {result.attempts}")
    print(f"query_style: {result.query_style}")
