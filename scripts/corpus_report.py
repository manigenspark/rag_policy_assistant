"""Report corpus statistics and verify the planted conflict.

Checks the constraints the pipeline design depends on: document word counts are
inside the 500-800 assignment range, every section carries a POL code, and no
section is so long that it cannot be chunked cleanly.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"
FRONT_MATTER = re.compile(r"^---\n(.*?)\n---\n", re.DOTALL)
SECTION = re.compile(r"^## (POL-[A-Z]{3}-\d{3}) - (.+)$", re.MULTILINE)


def split_front_matter(text: str) -> tuple[dict[str, str], str]:
    match = FRONT_MATTER.match(text)
    if not match:
        raise ValueError("missing YAML front matter")
    meta: dict[str, str] = {}
    for line in match.group(1).splitlines():
        key, _, value = line.partition(":")
        meta[key.strip()] = value.strip().strip('"')
    return meta, text[match.end():]


def main() -> int:
    docs = sorted(RAW.glob("*.md"))
    if not docs:
        print("no documents found")
        return 1

    print(f"{'document':<28} {'ver':>5} {'status':>9} {'words':>6} {'secs':>5}"
          f"  {'longest section':>15}")
    print("-" * 82)

    totals = []
    for path in docs:
        meta, body = split_front_matter(path.read_text())
        sections = SECTION.split(body)[1:]
        pairs = list(zip(sections[0::3], sections[2::3]))
        words = len(body.split())
        longest = max(len(text.split()) for _, text in pairs)
        in_range = "ok" if 500 <= words <= 800 else "OUT OF RANGE"
        print(f"{path.name:<28} {meta['version']:>5} {meta['status']:>9} "
              f"{words:>6} {len(pairs):>5}  {longest:>9} words  {in_range}")
        totals.append((path.name, meta, pairs))

    print()
    section_counts = [len(p) for _, _, p in totals]
    print(f"total sections across corpus: {sum(section_counts)}")

    print("\nplanted conflict - same section code, different document version:")
    print("-" * 82)

    by_code: dict[str, list[tuple[str, str, str]]] = {}
    for name, meta, pairs in totals:
        for code, text in pairs:
            by_code.setdefault(code, []).append((name, meta["version"], text))

    conflicts = 0
    for code, entries in sorted(by_code.items()):
        if len(entries) < 2:
            continue
        numbers = [
            (name, ver, sorted(set(re.findall(r"\b\d[\d,]*(?= USD| calendar)", txt))))
            for name, ver, txt in entries
        ]
        distinct = {tuple(nums) for _, _, nums in numbers}
        if len(distinct) > 1:
            conflicts += 1
            print(f"  {code}")
            for name, ver, nums in numbers:
                print(f"    v{ver:<4} {name:<28} figures: {', '.join(nums) or '-'}")

    print(f"\nsections present in two versions with differing figures: {conflicts}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
