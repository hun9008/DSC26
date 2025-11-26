# src/util/logger.py

import os
import sys
import inspect
from datetime import datetime

class TeeLogger:
    """
    콘솔 출력 + 로그 파일 저장
    prefix=None이면 실행 중인 최상위 파일 이름을 자동으로 prefix로 사용
    """

    def __init__(self, log_dir="../logs", prefix=None):
        # prefix 자동 감지
        if prefix is None:
            # 최상위 호출 스택(프레임) 찾기
            frame = inspect.stack()[-1]
            script_path = frame.filename           
            base = os.path.basename(script_path)   
            prefix = os.path.splitext(base)[0]     

        self.prefix = prefix

        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_path = os.path.join(log_dir, f"{prefix}_{timestamp}.txt")

        # 파일 열기
        self.file = open(self.log_path, "w", encoding="utf-8")

        sys.__stdout__.write(f"[Logger] Logging to: {self.log_path}\n")

    def write(self, msg):
        sys.__stdout__.write(msg)
        self.file.write(msg)     

    def flush(self):
        self.file.flush()

    def close(self):
        self.file.close()