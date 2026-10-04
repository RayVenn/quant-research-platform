from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar

import pandas as pd


class DataProvider(ABC):
    """Fetches bars from an external source. Implement ``fetch`` to add a vendor.

    ``fetch`` returns one DataFrame per symbol with a DatetimeIndex and lowercase
    columns (``close`` required; ``open/high/low/volume`` optional). Prices should be
    split/dividend adjusted so returns are total returns.
    """

    name: ClassVar[str] = ""

    def __init__(self, **options):
        self.options = options

    @abstractmethod
    def fetch(
        self, symbols: list[str], start: str | None = None, end: str | None = None, interval: str = "1d"
    ) -> dict[str, pd.DataFrame]: ...
