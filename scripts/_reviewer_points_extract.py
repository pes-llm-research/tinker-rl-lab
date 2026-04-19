#!/usr/bin/env python3
"""Emit id|marker|severity rows from reviewer_points.yaml for the shell scorer."""
import re
import sys


def main(path: str) -> int:
    try:
        import yaml
        doc = yaml.safe_load(open(path))
        for w in doc.get("weaknesses", []):
            sev = w.get("severity", "medium")
            print(f"{w['id']}|{w['marker']}|{sev}")
        return 0
    except ImportError:
        pass
    except Exception as exc:
        print(f"yaml parse failed: {exc}", file=sys.stderr)

    # Regex fallback
    text = open(path).read()
    blocks = re.split(r"\n  - id:\s*", text)[1:]
    for b in blocks:
        mid = b.splitlines()[0].strip()
        mk = re.search(r'marker:\s*"([^"]+)"', b)
        sv = re.search(r"severity:\s*(\w+)", b)
        if mk:
            print(f"{mid}|{mk.group(1)}|{sv.group(1) if sv else 'medium'}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
