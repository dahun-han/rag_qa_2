"""
sampling_plan.py — config/sampling_plan.yaml 기반 슬롯 플랜 생성 (plan.md §2 구현)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Union

import pandas as pd
import yaml

# 후보 슬롯 풀 크기 산정용 상수 - build_sampling_plan()에서만 쓴다. 풀의 남는 행은
# 백필에 실제로 쓰이지 않으면 아무 비용도 들지 않으므로(§ build_sampling_plan 참고,
# main.py가 idx 포인터로 필요한 만큼만 꺼내 쓴다) 넉넉하게 고정해도 문제없다 -
# total_count에 따라 사용자가 직접 튜닝할 필요가 없다.
_POOL_OVERSAMPLE = 5      # 후보 슬롯 풀 = total_count의 몇 배로 만들지
_POOL_MIN_BUFFER = 200    # 최소 여유분(절대값). total_count가 작아도 이만큼은 더 만든다

# §4-1 근거 청크 개수 범위 - 난이도와 무관하게 3~5 고정, all_neg만 0(resolve_positive_chunk_count
# 참고). pool_oversample/pool_min_buffer와 같은 이유로 사용자 튜닝이 필요 없는 값이라
# yaml에서 빼고 상수로 고정했다.
_POSITIVE_CHUNK_COUNT_RANGE = (3, 5)


@dataclass
class SamplingConfig:
    total_count: int
    data_type_ratio: dict
    difficulty_ratio: dict
    category_ratio: dict
    persona_ratio: dict
    # 청킹방식별 비중 - total_count(전체 최종 목표 건수)를 세 청킹방식에 어떻게 나눌지.
    # 비중 0인 방식은 목표가 0이 되어 main.py가 그 방식의 산출물 자체를 만들지 않는다
    # (예: 특정 청킹방식만 따로 돌리고 싶을 때 나머지를 0으로 둔다).
    chunk_method_ratio: dict
    candidate_pool_size: dict
    retrieval_eval: dict           # retrieval.json/answer.json 모의 검색 top_k 범위 (top_k_min/top_k_max)
    # 청킹방식별 "이미 다른 배치에서 만들어져 눈으로 확인한 개수" - 이번 실행의 item_id가
    # 이 값 다음 번호부터 이어지게 한다(export.py의 ID_BASE와 함께 씀).
    # 자동으로 기존 산출물을 스캔해 이어받지 않는다 - 사용자가 실행 전 직접 확인해 채운다.
    start_seq: dict = field(default_factory=dict)


def load_sampling_config(path: str = "config/sampling_plan.yaml") -> SamplingConfig:
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return SamplingConfig(**raw)


def allocate(total: int, ratio: dict) -> dict:
    """비율(ratio)을 정수 개수로 배분한다. 최대잔여법으로 반올림 오차를 보정한다.
    ratio 값의 합으로 정규화하므로, difficulty_ratio(0.5/0.3/0.2)처럼 이미 합계 1인
    비율은 물론 data_type_ratio(8:1:1)처럼 합계가 1이 아닌 가중치를 그대로 줘도
    total을 정확히 나눈다 — 정규화 없이 total*v만 하면 ratio 합계가 1이 아닐 때
    배분 총합이 total의 배수로 부풀어(예: 8+1+1=10배) 목표 건수 자체가 어긋난다.
    main.py의 allocate_targets()도 이 함수를 그대로 재사용한다(사본 두지 말 것)."""
    ratio_sum = sum(ratio.values())
    raw = {k: total * v / ratio_sum for k, v in ratio.items()}
    floors = {k: int(math.floor(v)) for k, v in raw.items()}
    remainder = total - sum(floors.values())
    order = sorted(raw, key=lambda k: raw[k] - floors[k], reverse=True)
    for k in order[:remainder]:
        floors[k] += 1
    return floors


def allocate_targets(n_demo: int, data_type_ratio: dict, difficulty_ratio: dict) -> dict:
    """n_demo를 (data_type, difficulty) 조합별 목표 건수로 배분한다.
    data_type_ratio와 difficulty_ratio를 곱한 2단계 배분 -> pos_neg의 '상' 같은 조합도
    비율대로 목표가 잡혀서, 실패 없이도 특정 난이도가 누락되지 않는다.
    all_pos도 하/중/상 전 난이도를 만든다(§1-1) — difficulty_ratio는 data_type과
    무관하게 동일 적용된다."""
    targets: dict = {}
    for data_type, dt_n in allocate(n_demo, data_type_ratio).items():
        for difficulty, d_n in allocate(dt_n, difficulty_ratio).items():
            targets[(data_type, difficulty)] = d_n
    return targets


def allocate_targets_by_category(
    n_demo: int, category_ratio: dict, data_type_ratio: dict, difficulty_ratio: dict
) -> dict[tuple[str, str, str], int]:
    """n_demo를 (카테고리, data_type, 난이도) 조합별 목표 건수로 배분한다.
    allocate_targets()에 카테고리 축을 한 단계 더 얹은 버전 — 카테고리마다 동일한
    data_type×난이도 비율을 적용해, main.py의 1단계(base 생성) 백필과 2단계(청킹방식별
    사후 백필)가 카테고리 단위로 정확한 목표를 참조할 수 있게 한다."""
    targets: dict[tuple[str, str, str], int] = {}
    for category, c_n in allocate(n_demo, category_ratio).items():
        for (data_type, difficulty), dd_n in allocate_targets(c_n, data_type_ratio, difficulty_ratio).items():
            targets[(category, data_type, difficulty)] = dd_n
    return targets


def _base_sort_key(cfg: SamplingConfig):
    """최종 저장 직전 base를 재정렬할 때 쓰는 정렬 키를 만든다: 카테고리(업무영역) →
    data_type → 난이도 → persona 순. 각 축의 우선순위는 config/sampling_plan.yaml에
    그 축의 ratio dict가 선언된 순서를 그대로 따른다 (예: category_ratio가
    LS/DS/FX/CD/WM/CR 순으로 적혀있으면 그 순서). 완성 순서(스레드 동시 처리라
    비결정적)로 번호가 매겨지던 문제를 없애, 실행마다 같은 질문이 같은 ID를 받게 한다."""
    category_order = {c: i for i, c in enumerate(cfg.category_ratio.keys())}
    data_type_order = {d: i for i, d in enumerate(cfg.data_type_ratio.keys())}
    difficulty_order = {d: i for i, d in enumerate(cfg.difficulty_ratio.keys())}
    persona_order = {p: i for i, p in enumerate(cfg.persona_ratio.keys())}

    def key(base: dict) -> tuple:
        return (
            category_order.get(base["category"], 999),
            data_type_order.get(base["data_type"], 999),
            difficulty_order.get(base["difficulty"], 999),
            persona_order.get(base["persona"], 999),
            base["index"],  # 같은 조합 내에서는 원래 슬롯 index로 안정 정렬
        )

    return key


def build_sampling_plan(cfg: SamplingConfig) -> pd.DataFrame:
    """
    data_type × difficulty × category × persona 조합별 목표 건수를 산출해
    슬롯 테이블을 만든다. 각 행이 이후 §4-1(positive 근거 선정) ~ Q0(질의 생성
    입력 조립)의 출발점이 된다.

    ★ 슬롯 "풀" 크기는 total_count가 아니라 pool_size(= total_count보다 큼)로 만든다.
    total_count는 "실제 목표 생성 건수"이고, 풀은 실패한 슬롯을 대체할 백필용
    후보까지 포함한 "후보 슬롯 전체"다. 둘이 같으면 조합 하나당 후보가 목표 개수만큼만
    있어서, 그 후보가 실패하는 순간 대체할 게 하나도 없어진다(특히 total_count를 작게
    낮춰 테스트할 때 두드러짐 — 조합당 후보가 0~1개뿐이라 백필이 사실상 불가능해짐).

    ★ all_pos도 '하'를 포함한 하/중/상 전 난이도를 만든다(§1-1) — data_type과
    difficulty는 독립된 두 축이라 difficulty_ratio는 data_type과 무관하게 동일 적용된다.
    """
    pool_size = max(cfg.total_count * _POOL_OVERSAMPLE, cfg.total_count + _POOL_MIN_BUFFER)

    rows = []
    seq_by_category: dict[str, int] = {}

    for data_type, dt_n in allocate(pool_size, cfg.data_type_ratio).items():
        for difficulty, d_n in allocate(dt_n, cfg.difficulty_ratio).items():
            for category, c_n in allocate(d_n, cfg.category_ratio).items():
                for persona, p_n in allocate(c_n, cfg.persona_ratio).items():
                    for _ in range(p_n):
                        seq_by_category[category] = seq_by_category.get(category, 0) + 1
                        rows.append({
                            "index": f"{category}_{seq_by_category[category]:03d}",
                            "data_type": data_type,
                            "difficulty": difficulty,
                            "category": category,
                            "target_persona": persona,
                        })
    return pd.DataFrame(rows)


def resolve_positive_chunk_count(data_type: str) -> Union[int, tuple[int, int]]:
    """§4-1에서 이 슬롯에 몇 개의 positive 청크를 골라야 하는지 조회.
    근거 개수는 난이도와 무관하게 3~5 고정(_POSITIVE_CHUNK_COUNT_RANGE). all_neg만 0.
    난이도(하/중/상) 구분은 근거 개수가 아니라 main.py의 pick_positive_chunks()가
    담당하는 '분산'(한 조항/여러 조항/복수 문서)으로 이뤄진다."""
    if data_type == "all_neg":
        return 0
    return _POSITIVE_CHUNK_COUNT_RANGE


if __name__ == "__main__":
    cfg = load_sampling_config("config/sampling_plan.yaml")
    plan = build_sampling_plan(cfg)
    print(plan.shape)
    print(plan.head(10))
    print(plan["data_type"].value_counts())
    print(plan["difficulty"].value_counts())
    print(plan["category"].value_counts())
    print(plan["target_persona"].value_counts())
    print("positive_chunk_count 예시:", resolve_positive_chunk_count("pos_neg"))
