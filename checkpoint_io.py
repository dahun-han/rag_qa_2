"""checkpoint_io.py — 체크포인트 저장(엑셀/JSON append, 재시도, 이전 결과 정리).

"이번 배치에서 새로 생긴 항목을 어떻게 디스크에 남기는가"만 다루고, 어떤 항목을
만들지(LLM 호출 등)는 전혀 모른다.
"""

from __future__ import annotations

import logging
import os
import shutil
import time

from export import build_answer_records, build_retrieval_records, save_excel, save_json_items
from retrieval_sim import RetrievalSim, simulate_retrieval

logger = logging.getLogger("main")


def _save_with_retry(fn, *args, max_retries: int = 5, backoff: float = 1.5, **kwargs) -> None:
    """체크포인트 저장 중 PermissionError(Windows에서 OneDrive 동기화·백신 실시간 검사·
    파일을 다른 프로그램에서 열어둔 경우 등으로 순간적으로 파일이 잠겨 있을 때 발생)를
    몇 번 재시도한 뒤에도 안 풀리면 그때 최종 실패로 처리한다."""
    last_err: PermissionError | None = None
    for attempt in range(1, max_retries + 1):
        try:
            fn(*args, **kwargs)
            return
        except PermissionError as e:
            last_err = e
            if attempt < max_retries:
                time.sleep(backoff * attempt)
    raise RuntimeError(f"{max_retries}회 재시도 후에도 파일 저장 실패(다른 프로그램이 파일을 열어두진 않았는지 확인): {last_err}") from last_err


def build_retrieval_sims(items: list, cfg, cache: dict[str, RetrievalSim]) -> dict[str, RetrievalSim]:
    """문항별로 한 번만 모의 검색을 돌려 retrieval.json/answer.json이 같은
    top_k·같은 재분류 data_type을 공유하게 한다(따로 계산하면 랜덤 시드는 같아도
    두 파일을 만드는 시점의 candidate_pool 순서 등이 어긋날 위험이 있다).
    cache는 main()이 청킹방식별로 들고 있다가 매 체크포인트 호출마다 그대로 넘기는
    누적 딕셔너리다. items는 이번 체크포인트에서 새로 추가된 항목만 들어오므로
    보통은 전부 새 index지만, 재시도 등으로 우연히 겹치는 경우에 대비해 이미 있는
    index는 건너뛰고 새 문항만 계산해 cache에 채워 넣는다."""
    for it in items:
        if it.index not in cache:
            cache[it.index] = simulate_retrieval(
                it.index, it.query, it.candidate_pool,
                cfg.retrieval_eval["top_k_min"], cfg.retrieval_eval["top_k_max"],
            )
    return cache


def save_checkpoint(
    chunk_pool_by_method: dict, cfg, new_items_by_method: dict,
    sims_by_method: dict, output_dir: str, include_input_data: bool = False, as_of_date: str | None = None,
) -> None:
    """이번 체크포인트에서 새로 추가된 항목(new_items_by_method, 청킹방식별)만 파일에
    이어서 저장한다. retrieval/answer는 항목(index)당 파일 1개, 엑셀은 기존 워크북을
    열어 새 행만 append한다(save_excel 참고) — 이전 체크포인트에서 이미 쓴 내용은
    다시 읽거나 재작성하지 않는다. 이 폴더는 OneDrive 동기화 대상이라(_save_with_retry의
    PermissionError 재시도가 그 증거), 안 바뀐 내용을 매번 다시 쓰면 OneDrive 재업로드로
    잠금 경합이 커진다 — 새 항목만 쓰면 이 문제가 없다.

    include_input_data=False(체크포인트 중간 저장 기본값)면 INPUT_DATA 시트를 뺀다 —
    이 시트는 items와 무관하게 항상 같은 내용(입력 파일 전체 청크)이라 최종 저장
    (main()에서 generate_items 종료 후 1회) 때만 include_input_data=True로 채운다."""
    for method in chunk_pool_by_method:
        new_items = new_items_by_method[method]
        if new_items or include_input_data:  # 새 항목도 없고 INPUT_DATA 채울 것도 아니면 파일을 아예 안 건드림
            _save_with_retry(
                save_excel, new_items, system_nm_lookup={}, out_path=f"{output_dir}/output_{method}.xlsx",
                input_chunks=chunk_pool_by_method[method] if include_input_data else None, as_of_date=as_of_date,
            )
        if not new_items:
            continue
        sims = build_retrieval_sims(new_items, cfg, sims_by_method[method])
        # 청킹방식 3개 × 문항 100개 = 300건이 각각 고유 ID·고유 파일을 갖도록, 방식별로
        # 나누지 않고 하나의 retrieval/ , answer/ 디렉터리에 함께 저장한다 (item_id의
        # 번호 구간 자체가 청킹방식을 구분해주므로 파일명이 겹치지 않는다).
        _save_with_retry(
            save_json_items, build_retrieval_records(new_items, sims), f"{output_dir}/retrieval",
        )
        _save_with_retry(
            save_json_items, build_answer_records(new_items, sims), f"{output_dir}/answer",
        )


def _clear_output_files(output_dir: str, chunk_pool_by_method: dict) -> None:
    """완성 순서 기준 번호로 이미 저장된 retrieval/answer json과 청킹방식별 엑셀을
    지운다 - 정렬된 새 seq로 다시 저장하기 위한 선행 작업(재번호 매김 전 파일이
    새 파일과 뒤섞이면 번호가 겹치거나 옛 파일이 잔류할 수 있다)."""
    for sub in ("retrieval", "answer"):
        d = f"{output_dir}/{sub}"
        if os.path.isdir(d):
            shutil.rmtree(d)
    for method in chunk_pool_by_method:
        p = f"{output_dir}/output_{method}.xlsx"
        if os.path.exists(p):
            os.remove(p)
