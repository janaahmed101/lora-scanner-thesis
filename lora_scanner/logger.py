from __future__ import annotations

from pathlib import Path
import json
import sys
from typing import Any


class JsonlLogger:
    def __init__(self, path: str | Path | None):
        self.path = Path(path) if path else None
        self.path.parent.mkdir(parents=True, exist_ok=True) if self.path else None
        self._fh = self.path.open("a", encoding="utf-8") if self.path else None

    def write(self, record: dict[str, Any]) -> None:
        line = json.dumps(record, ensure_ascii=False, sort_keys=True, default=str)
        if self._fh:
            self._fh.write(line + "\n")
            self._fh.flush()
        print(line, file=sys.stdout)

    def close(self) -> None:
        if self._fh:
            self._fh.close()
