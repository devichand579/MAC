from typing import List
# cache with imits
from functools import lru_cache


@lru_cache(maxsize=1000)
def longest_Common_Prefix(list_strs: List[str]) -> str:

    if not list_strs:
        return ""

    short_str = min(list_strs, key=len)

    for i, char in enumerate(short_str):
        for other in list_strs:
            if other[i] != char:
                return short_str[:i]

    return short_str
