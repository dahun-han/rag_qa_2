"""llm_config.py — 사용할 GPT 모델과 모델별 파라미터를 config에서 로드"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import yaml


@dataclass
class LLMConfig:
    model: str = "gpt-5-mini"
    supports_temperature: bool = False   # false면 temperature를 API에 안 보냄
    temperature: float = 1.0
    max_tokens: Optional[int] = None     # 응답 토큰 한도(추론 토큰 포함). None=API 기본값
    max_workers: int = 4                 # 동시 처리 슬롯 수
    checkpoint_every: int = 5            # N건마다 중간 저장
    qa_retry: int = 3                    # QA 검증 실패 시 답변/Nugget/reason 재생성 최대 횟수
    query_max_attempts: int = 3          # 질문 생성 재시도 최대 횟수
    query_batch_size: int = 3            # 한 라운드에 만들 질문 후보 개수
    query_max_tokens: Optional[int] = None  # 질문 후보 배치 생성 콜 전용 토큰 한도(None=전역 max_tokens)


def load_llm_config(path: str = "config/llm_config.yaml") -> LLMConfig:
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return LLMConfig(**raw)
