"""main.py — 전체 파이프라인 오케스트레이터. 모델/파라미터는 config/llm_config.yaml에서 관리한다."""

from __future__ import annotations

import os
import random
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime

from tqdm import tqdm

from answer_gen import analyze_answer, generate_ground_truth
from candidate_pool import build_candidate_pool
from checkpoint_io import _clear_output_files, save_checkpoint
from chunk_pool import load_chunk_pool
from chunk_selection import (
    DocCandidate,
    build_category_cache,
    build_document_cache,
    classify_difficulty,
    pick_excluded_doc,
    pick_positive_chunks,
    prune_to_minimal_evidence,
)
from export import (
    QAItem,
    format_kr_date,
    resolve_start_seq,
    save_answer_summary_xlsx,
    save_input_chunks_xlsx,
    save_merged_qa_xlsx,
    save_report_xlsx,
    save_retrieval_summary_xlsx,
)
from llm_config import load_llm_config
from logging_setup import _attach_file_logging, logger
from query_pipeline import (
    OpenAILLMClient,
    QueryGenConfig,
    QueryGenerationFailed,
    QueryGenerator,
    QuerySlot,
)
from reason_gen import generate_reason
from retag import CATEGORY_CODE, CODE_TO_CATEGORY
from retrieval_sim import RetrievalSim
from sampling_plan import (
    _base_sort_key,
    allocate,
    allocate_targets_by_category,
    build_sampling_plan,
    load_sampling_config,
    resolve_positive_chunk_count,
)
from timing import reset as reset_timing, summary as timing_summary
from validation import validate_query_answer, validate_structure

INPUT_PATHS = {
    "recursive_300_100": "input/recursive_300_100_chunk_span.xlsx",
    "recursive_600_200": "input/recursive_600_200_chunk_span.xlsx",
    "recursive_1000_300": "input/recursive_1000_300_chunk_span.xlsx",
}
OUTPUT_DIR = "output"


def build_item(
    slot_row, cfg, candidates: list, doc_candidates: list[DocCandidate], chunking_method: str, llm,
    qa_retry: int = 3, query_max_attempts: int = 3, query_batch_size: int = 3, query_max_tokens: int | None = None,
) -> QAItem | None:
    """슬롯(카테고리·data_type·난이도·persona 목표) 하나와 청킹방식 하나를 받아, 질문
    생성부터 답변·Nugget·근거 청크·reason·candidate_pool까지 전부 그 청킹방식의 청크
    풀만으로 독립적으로 만든다. 세 청킹방식은 같은 슬롯을 공유할 뿐, 실제로 어느 문서·
    청크를 근거로 쓸지와 질문 문장 자체는 방식마다 완전히 따로 생성된다 — 청크 크기
    차이 자체가 만드는 질문·근거 다양성도 벤치마크가 관찰하려는 대상의 일부이기 때문이다.
    candidates/doc_candidates: build_category_cache()/build_document_cache()로 미리
    계산한 (chunking_method, category) 후보."""
    if not candidates or not doc_candidates:
        return None

    n_spec = resolve_positive_chunk_count(slot_row["data_type"])

    # 슬롯 index만으로 시드를 잡으면 세 청킹방식이 같은 문서·같은 청크를 뽑아 결국 질문도
    # 비슷해지기 쉽다(문서 후보 목록 자체는 방식과 무관하게 같은 순서로 시작하므로). 방식을
    # 시드에 섞어야 방식별로 독립적인 무작위 선택이 보장된다.
    item_seed = hash((slot_row["index"], chunking_method)) % (2**31)
    rng = random.Random(item_seed)

    n = rng.randint(n_spec[0], n_spec[1]) if isinstance(n_spec, tuple) else n_spec
    # all_pos-하는 정의상 "Positive에 Noise가 없어야"(=1개) 한다(§1-1). pos_neg-하처럼
    # 3~5개를 넉넉히 뽑고 나중에 essential_chunk_ids로 줄이면, 최종 결과가 우연히
    # 1개로 수렴했을 때 pos_neg-하와 과정 자체가 구분이 안 된다(뽑아놓고 버린 게
    # "노이즈"인지 아닌지 알 길이 없음). all_pos-하는 애초에 소재 선정 단계부터
    # 정확히 1개만 뽑아서, "버릴 노이즈 자체가 없는" 상태를 구조적으로 보장한다.
    if slot_row["data_type"] == "all_pos" and slot_row["difficulty"] == "하":
        n = 1

    excluded_doc_nm = None
    excluded_doc_id = None
    context_chunks: list = []
    if slot_row["data_type"] == "all_neg":
        source_chunks = []
        excluded_chunks = pick_excluded_doc(candidates, doc_candidates, rng)
        excluded_doc_nm = excluded_chunks[0].doc_nm
        excluded_doc_id = excluded_chunks[0].doc_id
        context_chunks = excluded_chunks[:1]
    else:
        source_chunks = pick_positive_chunks(
            candidates, doc_candidates, max(n, 1), slot_row["difficulty"], rng, data_type=slot_row["data_type"]
        )

    # 질문/답변/Nugget/reason은 재시도마다 전부 다시 만든다 — collapse(복수 근거 요구
    # 조건 미충족) 같은 실패는 "질문 자체가 그 조건을 요구하도록 못 만들어졌다"는
    # 신호이므로, 답변만 다시 만들어봐야 같은 질문이면 또 collapse될 가능성이 높다.
    # 그래서 매 attempt마다 질문부터 새로 생성한다.
    ground_truth = nuggets = reason = q_result = None
    positive_chunks = positive_chunk_ids = positive_docs = doc_names = None
    difficulty_final = slot_row["difficulty"]
    source_text = "\n\n".join(c.content for c in source_chunks)
    for attempt in range(1, qa_retry + 1):
        attempt_start = time.monotonic()
        slot = QuerySlot(
            index=slot_row["index"], data_type=slot_row["data_type"], difficulty=slot_row["difficulty"],
            positive_chunks=source_chunks, target_persona=slot_row["target_persona"],
            excluded_doc_nm=excluded_doc_nm, context_chunks=context_chunks,
        )
        try:
            q_result = QueryGenerator(
                llm=llm, cfg=QueryGenConfig(
                    max_attempts=query_max_attempts, batch_size=query_batch_size, max_tokens=query_max_tokens,
                )
            ).generate(slot)
        except QueryGenerationFailed as e:
            logger.warning(
                "[%s/%s] attempt %d/%d 실패(%.1f초) - 질문 생성 실패: %s",
                slot_row["index"], chunking_method, attempt, qa_retry, time.monotonic() - attempt_start, e,
            )
            continue

        try:
            ground_truth = generate_ground_truth(llm, q_result.query, source_chunks)
            grounded, nuggets, essential_ids = analyze_answer(llm, ground_truth, source_chunks)
            # 1순위: LLM이 직접 알려준 essential_chunk_ids. 비어있으면(구형 클라이언트 등)
            # nugget value가 실제로 등장하는 청크만 남기는 텍스트 매칭으로 폴백.
            positive_chunks = (
                [c for c in source_chunks if c.chunk_id in essential_ids]
                or prune_to_minimal_evidence(source_chunks, nuggets)
            )
            positive_chunk_ids = [c.chunk_id for c in positive_chunks]
            positive_docs = {c.doc_id: c.doc_nm for c in positive_chunks}
            doc_names = list({c.doc_nm for c in positive_chunks}) or ([excluded_doc_nm] if excluded_doc_nm else [])

            requires_multi_doc = slot_row["difficulty"] == "상"  # pos_neg-상, all_pos-상 공통
            requires_multi_chunk = slot_row["difficulty"] in ("중", "상")  # 하는 data_type 무관 1개가 정상

            if requires_multi_doc and len({c.doc_id for c in source_chunks}) > 1:
                actual_doc_count = len({c.doc_id for c in positive_chunks})
                if actual_doc_count <= 1:
                    logger.warning(
                        "[%s/%s] attempt %d/%d 실패(%.1f초) - 근거가 복수 문서여야 하는데(%s) 1개 문서로 collapse됨 -> 질문부터 재생성",
                        slot_row["index"], chunking_method, attempt, qa_retry, time.monotonic() - attempt_start,
                        slot_row["data_type"] + "/" + slot_row["difficulty"],
                    )
                    continue

            if requires_multi_chunk and len(source_chunks) > 1 and len(positive_chunks) <= 1:
                logger.warning(
                    "[%s/%s] attempt %d/%d 실패(%.1f초) - 근거가 복수 조항(청크)이어야 하는데(%s) 1개로 collapse됨 -> 질문부터 재생성",
                    slot_row["index"], chunking_method, attempt, qa_retry, time.monotonic() - attempt_start,
                    slot_row["data_type"] + "/" + slot_row["difficulty"],
                )
                continue

            difficulty_final, n_topics, n_docs = classify_difficulty(positive_chunks, slot_row["difficulty"])

            if difficulty_final != slot_row["difficulty"]:
                logger.debug(
                    "[%s/%s] 목표 난이도 '%s' -> 실제 조항 수(%d)/문서 수(%d) 기준으로 '%s'로 재분류",
                    slot_row["index"], chunking_method, slot_row["difficulty"], n_topics, n_docs, difficulty_final,
                )

            reason_ctx = {
                "query": q_result.query, "ground_truth": ground_truth,
                "positive_chunk_ids": positive_chunk_ids, "doc_names": doc_names,
                "data_type": slot_row["data_type"], "difficulty": difficulty_final,
                "n_positive": len(positive_chunks), "n_topics": n_topics,
                "doc_spread": "단일 문서" if n_docs <= 1 else "복수 문서",
                "excluded_doc_nm": excluded_doc_nm,
            }
            reason = generate_reason(llm, reason_ctx)
        except Exception as e:  # noqa: BLE001 - API 일시 오류도 남은 재시도 안에서 흡수한다
            logger.warning(
                "[%s/%s] attempt %d/%d 실패(%.1f초) - 답변/Nugget 생성 중 오류: %s",
                slot_row["index"], chunking_method, attempt, qa_retry, time.monotonic() - attempt_start, e,
            )
            continue

        ok, failures = validate_query_answer(
            q_result.query, ground_truth, positive_chunk_ids, nuggets, reason, grounded, source_text
        )
        if ok:
            logger.info(
                "[%s/%s] attempt %d/%d 성공(%.1f초)",
                slot_row["index"], chunking_method, attempt, qa_retry, time.monotonic() - attempt_start,
            )
            break

        # 실패 사유가 전부 "reason에 근거 id 누락"뿐이면(질문/GT/nugget은 이미 검증 통과) 질문부터
        # 통째로 다시 만들 필요가 없다 — collapse처럼 질문 자체가 문제인 경우와 달리, reason
        # 누락은 서술 문장이 id를 빠뜨린 것뿐이라 reason만 다시 만들어도 나머지 결과는 그대로 유효하다.
        if failures and all(f.startswith("reason에 positive_chunk_ids") for f in failures):
            try:
                reason = generate_reason(llm, reason_ctx)
                ok, failures = validate_query_answer(
                    q_result.query, ground_truth, positive_chunk_ids, nuggets, reason, grounded, source_text
                )
                if ok:
                    logger.info(
                        "[%s/%s] attempt %d/%d 성공(%.1f초, reason 단독 재시도)",
                        slot_row["index"], chunking_method, attempt, qa_retry, time.monotonic() - attempt_start,
                    )
                    break
            except Exception as e:  # noqa: BLE001
                logger.warning("[%s/%s] reason 단독 재시도 실패: %s", slot_row["index"], chunking_method, e)

        logger.warning(
            "[%s/%s] attempt %d/%d 실패(%.1f초) - QA 품질 검증 실패: %s",
            slot_row["index"], chunking_method, attempt, qa_retry, time.monotonic() - attempt_start, failures,
        )
    else:
        logger.warning(
            "[%s/%s] %d회 재생성해도 QA 품질 검증 실패 - 이 슬롯은 제외", slot_row["index"], chunking_method, qa_retry,
        )
        return None

    pool_size_cfg = cfg.candidate_pool_size[slot_row["data_type"]]
    pool_seed = hash((slot_row["index"], chunking_method, "pool")) % (2**31)
    # all_neg는 positive_chunks가 항상 비어 있어서 build_candidate_pool 내부의 "positive
    # 문서 제외" 로직이 아무것도 걸러내지 못한다 - 실제로는 정답이 있지만 의도적으로 뺀
    # 문서(excluded_doc_id)가 여기서 안 걸러지면 candidate pool에 도로 섞여 들어가
    # "근거 없음" 조건이 깨진다. 그 문서의 청크는 후보에서 명시적으로 뺀다.
    candidate_source = candidates
    if slot_row["data_type"] == "all_neg" and excluded_doc_id:
        candidate_source = [c for c in candidates if c.doc_id != excluded_doc_id]
    candidate_pool = build_candidate_pool(
        slot_row["data_type"], positive_chunks, candidate_source, pool_size_cfg,
        query=q_result.query, seed=pool_seed,
    )

    ok, failures = validate_structure(positive_chunk_ids, candidate_pool, positive_docs)
    if not ok:
        logger.warning("[%s/%s] 구조 검증 실패: %s", slot_row["index"], chunking_method, failures)
        return None

    representative = positive_chunks[0] if positive_chunks else candidates[0]
    return QAItem(
        index=slot_row["index"], seq=0,  # 임시값 - generate_items()가 완성 순서로, main()이 최종 순서로 다시 매김
        chunking_method=chunking_method, data_type=slot_row["data_type"], difficulty=difficulty_final,
        persona=q_result.persona, category=CODE_TO_CATEGORY[slot_row["category"]],
        subcategory=representative.subcategory, query=q_result.query, positive_docs=positive_docs,
        candidate_pool=candidate_pool, positive_chunk_ids=positive_chunk_ids,
        ground_truth=ground_truth, nuggets=nuggets, reason=reason,
        positive_chunk_meta={c.chunk_id: c.content for c in positive_chunks},
    )


def _expand_targets_by_method(cfg, methods: list[str]) -> dict[tuple[str, str, str, str], int]:
    """cfg.total_count를 청킹방식별 (카테고리, data_type, 난이도) 목표로 확장한다.

    반드시 total_count를 chunk_method_ratio로 먼저 나누고(방식별 총량), 그 총량 각각에
    대해 category×data_type×난이도 배분(allocate_targets_by_category)을 독립적으로
    다시 돌리는 순서로 계산해야 한다 — 거꾸로 이미 잘게 쪼개진 (카테고리,data_type,난이도)
    목표 하나하나를 방식별로 allocate()하면, 그 값이 작을 때(흔한 경우: total_count가
    카테고리·유형·난이도 조합 수만큼 잘게 나뉘어 1건 안팎이 되는 상황) 최대잔여법의
    나머지 배정이 매번 ratio dict의 첫 번째 방식으로만 쏠려버린다(예: chunk_method_ratio가
    1:1:1이어도 결과가 20:0:0이 되는 식). total_count 레벨에서 먼저 나누면 숫자가 커서
    이 쏠림이 사실상 사라진다.

    비중 0(또는 배분 결과 0)인 방식은 결과 dict에 키 자체가 없다 - main()이 이 키
    집합으로 "이 방식은 산출물을 아예 안 만든다"를 판단한다(_active_methods 참고)."""
    expanded: dict[tuple[str, str, str, str], int] = {}
    for method, method_total in allocate(cfg.total_count, cfg.chunk_method_ratio).items():
        if method not in methods or method_total <= 0:
            continue
        method_targets = allocate_targets_by_category(
            method_total, cfg.category_ratio, cfg.data_type_ratio, cfg.difficulty_ratio,
        )
        for (category, data_type, difficulty), n in method_targets.items():
            if n > 0:
                expanded[(category, data_type, difficulty, method)] = n
    return expanded


def _active_methods(methods: list[str], chunk_method_ratio: dict) -> list[str]:
    """비중이 0보다 큰 청킹방식만 남긴다. 비중 0인 방식은 문항을 하나도 만들지 않으므로
    main()이 이 목록으로 최종 산출물(input_chunks.xlsx 시트, retrieval/answer json 등)
    저장 시 그 방식을 아예 건너뛴다."""
    return [m for m in methods if chunk_method_ratio.get(m, 0) > 0]


def _build_combos(
    targets: dict[tuple[str, str, str, str], int], plan_df,
) -> dict[tuple[str, str, str, str], dict]:
    """targets((카테고리, data_type, 난이도, 청킹방식) -> 목표건수)와 plan_df로부터
    generate_items()가 쓸 초기 combos 상태(조합별 후보 큐·진행 카운터)를 만든다. 같은
    (카테고리, data_type, 난이도)는 청킹방식이 달라도 같은 슬롯 풀(plan_df 큐)을 쓰지만,
    idx/got/in_flight/fails는 방식별로 독립적으로 추적한다 — 한 방식에서 앞쪽 슬롯 몇 개가
    검증 실패로 건너뛰어져도 다른 방식은 그 슬롯으로 성공할 수 있으므로, 진행 상황을
    공유하면 안 된다. 순수 함수로 분리해 스레드/LLM 호출 없이 단위 테스트 가능하게 한다."""
    combos: dict[tuple[str, str, str, str], dict] = {}
    for (category, data_type, difficulty, method), target_n in targets.items():
        if target_n <= 0:
            continue
        queue = plan_df[
            (plan_df["category"] == category)
            & (plan_df["data_type"] == data_type)
            & (plan_df["difficulty"] == difficulty)
        ].to_dict("records")
        combos[(category, data_type, difficulty, method)] = {
            "queue": queue, "idx": 0, "got": 0, "in_flight": 0, "target": target_n, "fails": 0,
        }
    return combos


# 콤보 하나가 계속 실패하면(예: target이 1이라 원래는 한 번에 한 슬롯씩만 순차 시도) 매
# 시도가 LLM 왕복(질문 생성+judge 등, 수십~백여 초)만큼 그대로 지연으로 쌓인다 - 실패가
# 쌓일수록 같은 콤보에서 여러 슬롯을 동시에 시도하게 해, 어차피 다른 콤보들이 이미 끝나
# max_workers가 노는 상황에서 그 여유를 이 콤보에 쓰게 한다. 최대 3개까지만 더 허용하는
# 이유: 무한정 허용하면(특히 target이 큰 콤보에서) 이미 목표를 채웠는데도 뒤늦게 도착하는
# 여분의 API 호출(비용) 낭비가 커진다 — _max_in_flight가 반환한 여유분을 넘겨 도착한 성공은
# generate_items()가 state["got"] < state["target"] 체크로 어차피 버리므로 정합성은 깨지지
# 않고, 비용과 지연 사이의 트레이드오프만 조절한다.
_STUCK_COMBO_EXTRA_CONCURRENCY = 3


def _max_in_flight(state: dict) -> int:
    """이 콤보가 지금 동시에 시도해도 되는 슬롯 개수(대기 중 futures 개수 상한).
    기본은 아직 못 채운 목표 개수(target - got)만큼만 - 이미 충분히 제출된 콤보에 더
    제출하지 않는다. 실패(fails)가 쌓일수록 그 위에 최대 _STUCK_COMBO_EXTRA_CONCURRENCY개까지
    여유를 더 준다."""
    remaining = state["target"] - state["got"]
    if remaining <= 0:
        return 0
    return remaining + min(state["fails"], _STUCK_COMBO_EXTRA_CONCURRENCY)


def generate_items(
    category_cache: dict, document_cache: dict, plan_df, methods: list[str], cfg, llm, llm_cfg, on_batch=None,
) -> list[QAItem]:
    """질문·답변·Nugget·근거 청크(비용이 큰 부분)를 (슬롯, 청킹방식) 조합마다 전부
    독립적으로 생성한다. 실패한 조합은 같은 (카테고리, data_type, 난이도, 청킹방식)
    조합의 다른 슬롯으로 자동 대체하며 목표 개수를 채운다.

    ★ 모든 (카테고리, data_type, 난이도, 청킹방식) 조합을 처음부터 동시에 채운다(라운드로빈
    제출) — 조합을 순차 처리하면 목표가 작은 조합에서 max_workers만큼의 스레드를 다 못
    쓰고 논다. combo별 in_flight 카운트를 target까지만 채우도록 기본 제한하되(§ _max_in_flight),
    실패가 쌓인 콤보는 그 위로 여분을 더 허용한다 - target이 1인 콤보(총 건수가 적을 때
    흔함)는 원래 슬롯 하나씩만 순차 시도해서, judge가 계속 거부하는 콤보 하나가 다른 콤보는
    다 끝난 뒤에도 혼자 수십 분씩 순차로 슬롯을 태우는 문제가 있었다 - 실패할수록 여러
    슬롯을 동시에 시도해 지연을 줄인다(그만큼 목표를 채운 뒤 뒤늦게 도착하는 API 호출은
    버려지는 비용 낭비지만, 지연 대비 감수할 만하다).

    목표 건수는 cfg.total_count를 cfg.chunk_method_ratio·category_ratio·data_type_ratio·
    difficulty_ratio로 (카테고리, data_type, 난이도, 청킹방식) 단위까지 확장해 만든다
    (_expand_targets_by_method 참고). on_batch: llm_cfg.checkpoint_every개가 쌓일 때마다
    즉시 호출되는 콜백({청킹방식: [QAItem, ...]}를 받음). 이게 없으면 전체 목표를 다 채울
    때까지 결과 파일이 하나도 안 나온다 - on_batch로 즉시 저장을 시작시켜 체크포인트가
    훨씬 일찍부터 쌓이게 한다."""
    items: list[QAItem] = []
    pending_by_method: dict[str, list[QAItem]] = {m: [] for m in methods}
    pending_count = 0
    method_counts: dict[str, int] = {m: 0 for m in methods}  # tqdm postfix용 방식별 누적 완료 건수
    expanded_targets = _expand_targets_by_method(cfg, methods)
    total_target = sum(t for t in expanded_targets.values() if t > 0)
    pbar = tqdm(total=total_target, desc="질문/답변 생성(청킹방식별 독립)", unit="건")

    def flush_pending():
        nonlocal pending_count
        if on_batch and pending_count:
            # 방식마다 완전히 독립적으로 생성되다 보니 체크포인트 시점에 방식별 진행
            # 속도가 들쭉날쭉해서, 이번 배치에 새 항목이 없는 방식도 있을 수 있다.
            # save_checkpoint(그리고 그 안의 on_batch)는 chunk_pool_by_method의 모든
            # 방식이 키로 존재한다고 가정하므로, 빈 리스트라도 모든 방식을 다 넘겨야 한다.
            on_batch({m: list(v) for m, v in pending_by_method.items()})
            for v in pending_by_method.values():
                v.clear()
            pending_count = 0

    combos: dict[tuple, dict] = _build_combos(expanded_targets, plan_df)

    def run_round(executor: ThreadPoolExecutor, futures: dict) -> None:
        """combos에 현재 설정된 target까지 제출·수집한다. 백필 라운드에서 target이
        늘어난 조합만 다시 채우면 되므로, 매 라운드 재사용 가능하게 분리했다."""
        nonlocal pending_count

        def submit_one(key) -> bool:
            state = combos[key]
            if state["in_flight"] >= _max_in_flight(state) or state["idx"] >= len(state["queue"]):
                return False
            row = state["queue"][state["idx"]]
            state["idx"] += 1
            state["in_flight"] += 1
            category, data_type, difficulty, method = key
            candidates = category_cache.get((method, category), [])
            doc_candidates = document_cache.get((method, category), [])
            fut = executor.submit(
                build_item, row, cfg, candidates, doc_candidates, method, llm,
                llm_cfg.qa_retry, llm_cfg.query_max_attempts, llm_cfg.query_batch_size, llm_cfg.query_max_tokens,
            )
            futures[fut] = (key, row)
            return True

        def submit_round_robin(n: int) -> None:
            """아직 목표 미달인 조합들을 순환하며 최대 n개까지 제출한다 (조합 간에
            워커를 골고루 나눠 쓰게 함 - 조합 하나가 목표를 다 채워도 다른 조합이
            바로바로 그 자리를 이어받는다)."""
            submitted = 0
            keys = [k for k, s in combos.items() if s["in_flight"] < _max_in_flight(s)]
            i = 0
            while submitted < n and keys:
                key = keys[i % len(keys)]
                if submit_one(key):
                    submitted += 1
                    i += 1
                else:
                    keys.remove(key)
                    if not keys:
                        break

        submit_round_robin(llm_cfg.max_workers - len(futures))

        while futures:
            done, _ = wait(list(futures.keys()), return_when=FIRST_COMPLETED)
            for fut in done:
                key, row = futures.pop(fut)
                state = combos[key]
                state["in_flight"] -= 1
                _, _, _, method = key
                try:
                    item = fut.result()
                except Exception as e:  # noqa: BLE001 - 한 조합의 예외로 전체가 죽지 않게 한다
                    logger.warning("[%s/%s] 처리 중 예외 발생, 다른 슬롯으로 대체: %s", row["index"], method, e)
                    item = None
                if item is not None and state["got"] < state["target"]:
                    # 최종 출력 파일들에 노출되는 seq는 main()에서 방식별로 카테고리/data_type/
                    # 난이도/persona 순으로 다시 정렬해 새로 매기므로, 여기서의 순번은 임시값일 뿐이다.
                    item.seq = len(items) + 1
                    items.append(item)
                    pending_by_method[method].append(item)
                    pending_count += 1
                    state["got"] += 1
                    method_counts[method] += 1
                    pbar.set_postfix(method_counts, refresh=False)
                    pbar.update(1)
                    if pending_count >= llm_cfg.checkpoint_every:
                        flush_pending()
                elif item is None:
                    # 목표를 이미 채운 뒤 뒤늦게 도착한 여분의 성공(item is not None인데
                    # got >= target)은 실패가 아니라 그냥 버려지는 것뿐이므로 fails에 안 센다.
                    state["fails"] += 1
            # 방금 빈 자리만큼(조합 무관하게) 다시 채워 넣어 항상 max_workers를 꽉 채운다.
            submit_round_robin(llm_cfg.max_workers - len(futures))

    with ThreadPoolExecutor(max_workers=llm_cfg.max_workers) as executor:
        futures: dict = {}
        run_round(executor, futures)

        # ★ 백필: 특정 조합(카테고리×data_type×난이도×청킹방식)의 후보 슬롯이 검증 실패로
        # 소진되면, 그 조합만 미달로 끝나고 전체 건수가 total_target에 못 미치는 문제가
        # 있었다. 아직 후보가 남은 다른 조합들에 부족분을 "원래 목표 비율"대로 재배분해
        # 다시 채운다 - 그래야 개별 조합 비율도 최대한 지키면서 total_target(설정한 총
        # 건수)도 채울 수 있다. 그래도 못 채우면(모든 조합의 후보가 다 소진) 그때는
        # 정직하게 미달로 남기고 로그를 남긴다.
        backfill_round = 0
        max_backfill_rounds = 20
        while True:
            total_got = sum(s["got"] for s in combos.values())
            shortfall = total_target - total_got
            if shortfall <= 0:
                break
            open_combos = {k: s for k, s in combos.items() if s["idx"] < len(s["queue"])}
            if not open_combos:
                logger.warning(
                    "목표 %d건 중 %d건만 생성됨(%d건 부족) - 모든 조합의 후보 슬롯이 소진되어 "
                    "더 이상 백필할 수 없음",
                    total_target, total_got, shortfall,
                )
                break
            backfill_round += 1
            if backfill_round > max_backfill_rounds:
                logger.warning("백필 %d라운드 초과 - 중단 (남은 부족분 %d건)", max_backfill_rounds, shortfall)
                break
            weight_sum = sum(expanded_targets[k] for k in open_combos)
            extra = allocate(shortfall, {k: expanded_targets[k] / weight_sum for k in open_combos})
            for key, add_n in extra.items():
                if add_n > 0:
                    combos[key]["target"] += add_n
            run_round(executor, futures)

    for key, state in combos.items():
        if state["got"] < state["target"]:
            logger.warning(
                "[%s/%s/%s/%s] 목표 %d건(백필 포함) 중 %d건만 채움 (후보 슬롯 %d개 전부 소진)",
                key[0], key[1], key[2], key[3], state["target"], state["got"], len(state["queue"]),
            )

    flush_pending()  # 마지막 자투리(checkpoint_every 미만으로 남은 것)도 반드시 흘려보낸다
    pbar.close()
    return items


def _reorder_by_seq(items_by_method: dict) -> None:
    """청킹방식별 items 리스트를 item.seq 오름차순으로 제자리 정렬한다."""
    for items in items_by_method.values():
        items.sort(key=lambda it: it.seq)


def _item_sort_key(cfg):
    """QAItem을 최종 저장 직전(카테고리→data_type→난이도→persona 순)에 정렬할 때 쓰는
    키를 만든다. sampling_plan._base_sort_key와 동일한 규칙을 쓰되, QAItem.category는
    한글명(예: "여신")으로 저장돼 있어 category_ratio의 코드(예: "LS")와 형식이 다르므로
    CATEGORY_CODE로 역변환해서 넘긴다."""
    base_key_fn = _base_sort_key(cfg)

    def key(it: QAItem) -> tuple:
        return base_key_fn({
            "category": CATEGORY_CODE.get(it.category, it.category),
            "data_type": it.data_type,
            "difficulty": it.difficulty,
            "persona": it.persona,
            "index": it.index,
        })

    return key


def main(n_demo: int | None = None, input_paths: dict | None = None, output_dir: str | None = None):
    """input_paths/output_dir: 기본 INPUT_PATHS/OUTPUT_DIR을 건드리지 않고 일회성으로 다른
    입력 파일·출력 위치를 쓰고 싶을 때만 넘긴다 (예: topic_list 없는 원본 파일로 임시 실행)."""
    reset_timing()  # 이전 main() 실행(같은 프로세스에서 반복 호출되는 테스트 등)의 잔여 기록 제거
    cfg = load_sampling_config("config/sampling_plan.yaml")
    llm_cfg = load_llm_config("config/llm_config.yaml")
    plan_df = build_sampling_plan(cfg)
    print("[1/7] 입력 청크 로딩 중...")
    chunk_pool_by_method = load_chunk_pool(input_paths or INPUT_PATHS)
    methods = list(chunk_pool_by_method.keys())
    active_methods = _active_methods(methods, cfg.chunk_method_ratio)
    chunk_counts = ", ".join(f"{m}: {len(c)}건" for m, c in chunk_pool_by_method.items())
    print(f"[1/7] 입력 청크 로딩 완료 ({chunk_counts})")
    category_cache = build_category_cache(chunk_pool_by_method)
    document_cache = build_document_cache(category_cache)
    print("[2/7] 카테고리/문서 캐시 구성 완료")
    llm = OpenAILLMClient(cfg=llm_cfg)
    run_started_at = datetime.now()
    as_of_date = format_kr_date(run_started_at)  # QA 시트 as_of_date 컬럼 - 실행 내내 같은 값으로 고정
    # output_dir을 넘기지 않으면 실행 시각으로 폴더를 자동 생성한다(예: output/260805_143012).
    # 매번 폴더명을 직접 정하지 않아도 실행마다 결과가 겹치지 않게 분리되게 하기 위함.
    output_dir = output_dir or os.path.join(OUTPUT_DIR, run_started_at.strftime("%y%m%d_%H%M%S"))
    # 실행 중인 콘솔이 어느 output_dir을 대상으로 도는지 한눈에 보이게, 진행바보다
    # 먼저 찍는다 - 동시에 여러 실행을 띄우거나 나중에 다시 볼 때 헷갈리지 않도록.
    print(f"작업 대상 폴더 : {output_dir}")

    if n_demo is None:
        n_demo = cfg.total_count  # config/sampling_plan.yaml의 total_count가 곧 실제 생성 목표건수

    targets = allocate_targets_by_category(n_demo, cfg.category_ratio, cfg.data_type_ratio, cfg.difficulty_ratio)
    # 이번 실행의 output_dir을 만들기 전에 스캔해야 한다 - 먼저 만들고 나면 방금 생긴 빈
    # 폴더가 이름순으로 "가장 최근"이 되어 자기 자신을 참조하게 된다(retrieval/answer가
    # 아직 하나도 없어 모든 방식이 0으로 잘못 리셋됨).
    start_seq = resolve_start_seq(OUTPUT_DIR)
    os.makedirs(output_dir, exist_ok=True)
    _attach_file_logging(output_dir)
    logger.info("작업 대상 폴더 : %s", output_dir)
    # n_demo를 그대로 찍지 않고 targets 실제 합계를 찍는다 — allocate()의 배분 결과가
    # 요청 건수와 어긋나면(설정 오류 등) 로그에서 바로 드러나야 하므로, 합계가 요청
    # 건수와 다르면 즉시 경고까지 남긴다.
    targets_sum = sum(targets.values())
    if targets_sum != n_demo:
        logger.warning(
            "목표 건수 배분 합계(%d)가 요청 건수(%d)와 다릅니다 — data_type_ratio/difficulty_ratio "
            "설정을 확인하세요(allocate()는 ratio 합계로 정규화하지만, 예상 밖의 값이면 여기서 어긋남).",
            targets_sum, n_demo,
        )
    logger.info(
        "목표 건수(카테고리×data_type×난이도별): %s (합계 %d건, chunk_method_ratio %s로 "
        "청킹방식 %s에 배분) — 미달 조합은 자동 백필로 재분배됨",
        targets, targets_sum, cfg.chunk_method_ratio, active_methods,
    )

    items_by_method: dict[str, list[QAItem]] = {m: [] for m in methods}
    sims_by_method: dict[str, dict[str, RetrievalSim]] = {m: {} for m in methods}

    def on_batch(new_items_by_method: dict[str, list[QAItem]]) -> None:
        for method, new_items in new_items_by_method.items():
            items_by_method[method].extend(new_items)
        save_checkpoint(chunk_pool_by_method, cfg, new_items_by_method, sims_by_method, output_dir)
        cumulative = {m: len(items) for m, items in items_by_method.items()}
        logger.info("중간 저장: 이번 배치 반영 (누적: %s)", cumulative)
        logger.info("호출 종류별 소요시간(누적, 지금까지):\n%s", timing_summary())

    # 질문/답변/Nugget/근거는 (슬롯, 청킹방식) 조합마다 완전히 독립적으로 생성된다.
    # checkpoint_every개가 쌓일 때마다 on_batch가 즉시 저장하므로, 전체 목표를 다 채울
    # 때까지 기다리지 않고도 결과 파일이 중간중간 갱신된다.
    print(f"[3/7] 질문/답변 생성 시작 (전체 목표 {targets_sum}건, 청킹방식 {active_methods}에 배분)")
    items = generate_items(
        category_cache, document_cache, plan_df, methods, cfg, llm, llm_cfg, on_batch=on_batch,
    )
    print(f"[3/7] 질문/답변 생성 완료: 총 {len(items)}건")

    # ★ 최종 재정렬: 생성 중에는 스레드 완료 순서(비결정적)로 seq가 매겨졌으니, 다 끝난
    # 지금 청킹방식별로 카테고리→data_type→난이도→persona 순으로 다시 정렬해 seq를 새로
    # 매긴다 - 실행마다 같은 조건의 문항이 같은 자리(번호)에 모이고, 파일을 열었을 때도
    # 같은 업무영역·유형끼리 번호가 모여 보인다. 세 청킹방식은 문항을 공유하지 않으므로
    # (방식마다 독립 생성) 이 정렬·번호 매김도 방식별로 완전히 독립적으로 한다.
    # start_seq(위에서 output_dir 생성 전에 미리 스캔해둔, 가장 최근 실행 폴더 기준
    # 방식별 최대 seq)가 있으면 그만큼 이어서 번호를 매긴다.
    sort_key = _item_sort_key(cfg)
    for method, method_items in items_by_method.items():
        method_items.sort(key=sort_key)
        start = start_seq.get(method, 0)
        for i, it in enumerate(method_items):
            it.seq = start + i + 1
    _reorder_by_seq(items_by_method)  # 위 정렬로 이미 seq 오름차순이지만, 방어적으로 한 번 더 보장
    print("[4/7] 최종 정렬 완료 (카테고리→data_type→난이도→persona 순)")

    # 완성 순서 기준 번호로 이미 저장된 retrieval/answer json을 지우고, 정렬된 새 seq로
    # 전부 다시 저장한다(체크포인트의 "새 항목만 append" 방식과 달리 이번엔 전체를 새로
    # 쓴다). chunk_method_ratio가 0인 방식은 active_methods에서 빠져 있으므로, 아래
    # 저장 함수들에 그 방식을 아예 안 넘긴다 - 문항이 0건인 방식은 산출물 자체가 안 생긴다.
    active_chunk_pool = {m: chunk_pool_by_method[m] for m in active_methods}
    active_items_by_method = {m: items_by_method[m] for m in active_methods}
    active_sims_by_method = {m: sims_by_method[m] for m in active_methods}
    _clear_output_files(output_dir)
    save_checkpoint(active_chunk_pool, cfg, active_items_by_method, sims_by_method, output_dir)
    print("[5/7] retrieval/answer json 최종 저장 완료 (정렬된 seq 기준)")

    # 산출물 3종: report.xlsx(집계), 통합 QA(검수용), retrieval/answer 요약(문항당
    # candidate_chunks/context 원문 대신 개수만 훑어보는 용도) - 청킹방식별
    # output_{method}.xlsx는 더 이상 만들지 않는다(통합 QA/요약이 그 역할을 대신함).
    # 그 안에만 있던 INPUT_DATA(입력 전체 청크)는 input_chunks.xlsx로 옮겼다.
    save_input_chunks_xlsx(active_chunk_pool, f"{output_dir}/input_chunks.xlsx")
    save_report_xlsx(active_items_by_method, f"{output_dir}/report.xlsx")
    print("[6/7] input_chunks.xlsx / report.xlsx 저장 완료")

    # 세 청킹방식의 QA/USED_DOCS를 하나로 합친 통합본 - 검수자가 파일 하나만 열어도
    # 전체를 훑어볼 수 있게 매 실행마다 자동 생성한다.
    merged_name = f"output_qa_{run_started_at.strftime('%y%m%d_%H%M%S')}.xlsx"
    merged_path = f"{output_dir}/{merged_name}"
    save_merged_qa_xlsx(active_items_by_method, system_nm_lookup={}, out_path=merged_path, as_of_date=as_of_date)
    save_retrieval_summary_xlsx(active_items_by_method, f"{output_dir}/retrieval_summary.xlsx")
    save_answer_summary_xlsx(active_items_by_method, active_sims_by_method, f"{output_dir}/answer_summary.xlsx")
    print(f"[7/7] 통합 파일/retrieval_summary.xlsx/answer_summary.xlsx 저장 완료: {merged_path}")

    logger.info(
        "전체 완료: 생성 %d건, 최종 저장 건수: %s, report.xlsx/%s 저장됨",
        len(items), {m: len(its) for m, its in items_by_method.items()}, merged_name,
    )
    logger.info("호출 종류별 소요시간(전체 실행 기준):\n%s", timing_summary())
    final_counts = ", ".join(f"{m}: {len(its)}건" for m, its in items_by_method.items())
    print(f"[7/7] 전체 완료: 생성 {len(items)}건, 최종 저장 건수: ({final_counts})")


if __name__ == "__main__":
    main()  # n_demo 생략 시 config/sampling_plan.yaml의 total_count가 목표 건수가 됨
