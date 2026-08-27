"""
sampling_plan.py — config/sampling_plan.yaml 기반 슬롯 플랜 생성 (plan.md §2 구현)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Union

import pandas as pd
import yaml

from retag import CATEGORY_CODE

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
class TargetItem:
    """구체적인 생성 대상을 직접 타겟팅할 때 사용하는 설정 구조체"""
    category: str          # 카테고리 코드 (예: "LS", "DS", "FX", "CD", "WM", "CR")
    chunking_method: str   # 청킹 방식 (예: "recursive_300_100", "recursive_600_200", "recursive_1000_300")
    data_type: str         # 데이터 유형 (예: "pos_neg", "all_pos", "all_neg")
    difficulty: str        # 난이도 (예: "상", "중", "하")
    count: int             # 목표 생성 수량


@dataclass
class SamplingConfig:
    total_count: int = 0
    data_type_ratio: dict = field(default_factory=dict)
    difficulty_ratio: dict = field(default_factory=dict)
    category_ratio: dict = field(default_factory=dict)
    persona_ratio: dict = field(default_factory=lambda: {"고객": 0.5, "내부직원": 0.5})
    # 청킹방식별 비중 - total_count(전체 최종 목표 건수)를 세 청킹방식에 어떻게 나눌지.
    # 비중 0인 방식은 목표가 0이 되어 main.py가 그 방식의 산출물 자체를 만들지 않는다
    # (예: 특정 청킹방식만 따로 돌리고 싶을 때 나머지를 0으로 둔다).
    chunk_method_ratio: dict = field(default_factory=dict)
    candidate_pool_size: dict = field(default_factory=dict)
    retrieval_eval: dict = field(default_factory=dict)  # retrieval.json/answer.json 모의 검색 top_k 범위 (top_k_min/top_k_max)
    targets: list[TargetItem] | None = None             # 타겟 지정 생성 모드용 구체적 대상 목록


def load_sampling_config(path: str = "config/sampling_plan.yaml") -> SamplingConfig:
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    raw_targets = raw.pop("targets", None) or raw.pop("target_list", None) or raw.pop("explicit_targets", None)
    parsed_targets: list[TargetItem] | None = None
    if raw_targets and isinstance(raw_targets, list):
        parsed_targets = []
        for t in raw_targets:
            if not isinstance(t, dict):
                continue
            cat = t.get("category") or t.get("카테고리")
            method = t.get("chunking_method") or t.get("chunk_method") or t.get("청킹방식")
            dtype = t.get("data_type") or t.get("데이터유형") or t.get("데이터_유형")
            diff = t.get("difficulty") or t.get("난이도")
            cnt = (
                t.get("count")
                or t.get("target_count")
                or t.get("목표 개수")
                or t.get("목표_개수")
                or t.get("목표개수")
                or t.get("수량")
                or 0
            )

            if not (cat and method and dtype and diff):
                continue

            cat_str = str(cat).strip()
            cat_code = CATEGORY_CODE.get(cat_str, cat_str)

            method_str = str(method).strip()
            if method_str in {"300_100", "600_200", "1000_300"}:
                method_str = f"recursive_{method_str}"

            try:
                cnt_int = int(cnt)
            except (ValueError, TypeError):
                cnt_int = 0

            if cnt_int > 0:
                parsed_targets.append(
                    TargetItem(
                        category=cat_code,
                        chunking_method=method_str,
                        data_type=str(dtype).strip(),
                        difficulty=str(diff).strip(),
                        count=cnt_int,
                    )
                )

        if not parsed_targets:
            parsed_targets = None

    total_count = raw.get("total_count", 0)
    if parsed_targets and not total_count:
        total_count = sum(t.count for t in parsed_targets)

    default_candidate_pool_size = {
        "pos_neg": {"mode": "doc_count", "min": 50, "max": 50},
        "all_pos": {"mode": "doc_count", "min": 50, "max": 50},
        "all_neg": {"mode": "doc_count", "min": 50, "max": 50},
    }
    default_retrieval_eval = {"top_k_min": 5, "top_k_max": 5}
    default_persona_ratio = {"고객": 0.5, "내부직원": 0.5}

    return SamplingConfig(
        total_count=total_count,
        data_type_ratio=raw.get("data_type_ratio") or {},
        difficulty_ratio=raw.get("difficulty_ratio") or {},
        category_ratio=raw.get("category_ratio") or {},
        persona_ratio=raw.get("persona_ratio") or default_persona_ratio,
        chunk_method_ratio=raw.get("chunk_method_ratio") or {},
        candidate_pool_size=raw.get("candidate_pool_size") or default_candidate_pool_size,
        retrieval_eval=raw.get("retrieval_eval") or default_retrieval_eval,
        targets=parsed_targets,
    )


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
    default_cats = ["LS", "DS", "FX", "CD", "WM", "CR"]
    cat_keys = list(cfg.category_ratio.keys()) if cfg.category_ratio else default_cats
    category_order = {c: i for i, c in enumerate(cat_keys)}

    default_dtypes = ["pos_neg", "all_pos", "all_neg"]
    dtype_keys = list(cfg.data_type_ratio.keys()) if cfg.data_type_ratio else default_dtypes
    data_type_order = {d: i for i, d in enumerate(dtype_keys)}

    default_diffs = ["하", "중", "상"]
    diff_keys = list(cfg.difficulty_ratio.keys()) if cfg.difficulty_ratio else default_diffs
    difficulty_order = {d: i for i, d in enumerate(diff_keys)}

    default_personas = ["고객", "내부직원"]
    persona_keys = list(cfg.persona_ratio.keys()) if cfg.persona_ratio else default_personas
    persona_order = {p: i for i, p in enumerate(persona_keys)}

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

    1) 타겟 모드 (cfg.targets 지정 시):
       - 지정된 targets 목록의 각 (category, data_type, difficulty) 조합별로 필요한 슬롯 풀을 생성한다.
       - 백필 여유분을 확보하기 위해 pool_size = max(comb_count * _POOL_OVERSAMPLE, comb_count + _POOL_MIN_BUFFER)로 생성한다.
       - persona_ratio에 맞춰 target_persona를 배분한다.

    2) 비율 모드 (cfg.targets 미지정 시):
       - total_count 및 각 ratio 비율에 따라 슬롯 풀을 생성한다.

    ★ 슬롯 "풀" 크기는 total_count가 아니라 pool_size(= total_count보다 큼)로 만든다.
    total_count는 "실제 목표 생성 건수"이고, 풀은 실패한 슬롯을 대체할 백필용
    후보까지 포함한 "후보 슬롯 전체"다. 둘이 같으면 조합 하나당 후보가 목표 개수만큼만
    있어서, 그 후보가 실패하는 순간 대체할 게 하나도 없어진다.
    """
    rows = []
    seq_by_category: dict[str, int] = {}

    if cfg.targets:
        comb_targets: dict[tuple[str, str, str], int] = {}
        for t in cfg.targets:
            key = (t.category, t.data_type, t.difficulty)
            comb_targets[key] = comb_targets.get(key, 0) + t.count

        for (category, data_type, difficulty), comb_count in comb_targets.items():
            pool_size = max(comb_count * _POOL_OVERSAMPLE, comb_count + _POOL_MIN_BUFFER)
            persona_alloc = allocate(pool_size, cfg.persona_ratio)
            for persona, p_n in persona_alloc.items():
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

    pool_size = max(cfg.total_count * _POOL_OVERSAMPLE, cfg.total_count + _POOL_MIN_BUFFER)

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
