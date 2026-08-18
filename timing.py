"""timing.py — 호출 종류별(Query 생성/Judge/Persona/GT/Analyze/Reason) 소요시간 계측(진단용).
여러 슬롯이 스레드로 동시에 도는 파이프라인이라, 락으로 보호된 공유 누적 구조에 모은다."""

from __future__ import annotations

import threading
import time
from collections import defaultdict
from contextlib import contextmanager

_lock = threading.Lock()
_durations: dict[str, list[float]] = defaultdict(list)


@contextmanager
def timed(label: str):
    start = time.monotonic()
    try:
        yield
    finally:
        elapsed = time.monotonic() - start
        with _lock:
            _durations[label].append(elapsed)


def reset() -> None:
    with _lock:
        _durations.clear()


def summary() -> str:
    """라벨별 건수/합계/평균/중앙값/최대값을 합계(전체 실행 시간에서 차지하는 비중) 내림차순으로 정리한다."""
    with _lock:
        snapshot = {label: list(values) for label, values in _durations.items()}

    if not snapshot:
        return "(기록된 호출 없음)"

    rows = []
    for label, values in snapshot.items():
        n = len(values)
        total = sum(values)
        avg = total / n
        values_sorted = sorted(values)
        median = values_sorted[n // 2]
        rows.append((label, n, total, avg, median, max(values)))

    rows.sort(key=lambda r: r[2], reverse=True)
    lines = [f"{'라벨':<10}{'건수':>6}{'합계(초)':>12}{'평균(초)':>10}{'중앙값(초)':>12}{'최대(초)':>10}"]
    for label, n, total, avg, median, mx in rows:
        lines.append(f"{label:<10}{n:>6}{total:>12.1f}{avg:>10.1f}{median:>12.1f}{mx:>10.1f}")
    return "\n".join(lines)
