"""logging_setup.py — 파이프라인 공용 로거("main") 설정.

콘솔은 ERROR만 보이게 조용히 하고, 실행별 파일 핸들러(output_dir/pipeline.log)는
main()이 output_dir을 알게 된 시점에 붙인다. 다른 모듈(checkpoint_io.py 등)은
logging.getLogger("main")으로 이 로거를 그대로 다시 얻어 쓴다 - 싱글턴이라 여기서
붙인 핸들러가 그대로 보인다.
"""

from __future__ import annotations

import logging

logger = logging.getLogger("main")
# 실행 중에는 tqdm 진행바 2개(질문/답변 생성, 청킹방식별 매핑/저장)만 콘솔에 보이게
# 한다 - 슬롯별 재시도/실패 경고(예: 조합별 후보 소진)는 콘솔엔 안 찍히지만, logger
# 자체 레벨은 INFO로 열어두고 콘솔 핸들러만 ERROR로 걸러서, main()이 붙이는 파일
# 핸들러(_attach_file_logging)에는 그대로 남게 한다 - "왜 목표 건수보다 적게
# 나왔는지"를 나중에 output_dir/pipeline.log에서 확인할 수 있게 하기 위함이다.
logger.setLevel(logging.INFO)
logger.propagate = False
_console_handler = logging.StreamHandler()
_console_handler.setLevel(logging.ERROR)
_console_handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
logger.addHandler(_console_handler)
# 이 파이프라인이 쓰는 라이브러리들의 진행 로그도 조용히 시킨다.
for noisy in ("openai", "httpx", "httpcore", "urllib3"):
    logging.getLogger(noisy).setLevel(logging.WARNING)


def _attach_file_logging(output_dir: str) -> None:
    """이 실행(output_dir)의 조합별 미달·재시도 경고를 pipeline.log에 전부 남긴다.
    콘솔은 여전히 조용하지만(ERROR만), 나중에 "왜 목표 건수보다 적게 나왔는지"를
    이 파일에서 확인할 수 있다. main()을 노트북 등에서 여러 번 호출해도 이전 실행의
    FileHandler가 계속 쌓이지 않도록, 새로 붙이기 전에 이전 FileHandler는 뗀다."""
    for h in [h for h in logger.handlers if isinstance(h, logging.FileHandler)]:
        logger.removeHandler(h)
        h.close()
    fh = logging.FileHandler(f"{output_dir}/pipeline.log", encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(fh)
