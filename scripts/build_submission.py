#!/usr/bin/env python3
"""Build the NeurIPS 2026 D&B submission bundle.

Inputs assembled from the live repo:
  paper/main.pdf              -> submission/contents/paper.pdf
  paper/main_anon.pdf         -> submission/contents/paper_anon.pdf
  paper/ethics_wrapper.pdf    -> submission/contents/ethics_statement.pdf
  blind_review/tinker-rl-lab-anon.tar.gz -> submission/contents/code.tar.gz

Refreshes checksums.sha256 / MANIFEST.md, then zips the whole bundle.
"""
from __future__ import annotations

import hashlib
import shutil
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONTENTS = ROOT / "submission" / "contents"
ZIP_PATH = ROOT / "submission" / "neurips2026_tinker_rl_lab.zip"


SOURCES = [
    (ROOT / "paper" / "main.pdf", CONTENTS / "paper.pdf"),
    (ROOT / "paper" / "main_anon.pdf", CONTENTS / "paper_anon.pdf"),
    (ROOT / "paper" / "ethics_wrapper.pdf", CONTENTS / "ethics_statement.pdf"),
    (
        ROOT / "blind_review" / "tinker-rl-lab-anon.tar.gz",
        CONTENTS / "code.tar.gz",
    ),
]

# Non-binary bundle files that live in contents/ already (authored content).
TEXT_MEMBERS = [
    "REVIEWER_README.md",
    "data_statement.md",
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    CONTENTS.mkdir(parents=True, exist_ok=True)
    for src, dst in SOURCES:
        if not src.exists():
            raise SystemExit(f"missing source: {src}")
        shutil.copy2(src, dst)
        print(f"copied {src.relative_to(ROOT)} -> {dst.relative_to(ROOT)}")

    # Canonical order for checksums and manifest
    order = [
        "ethics_statement.pdf",
        "paper.pdf",
        "paper_anon.pdf",
        "code.tar.gz",
        "REVIEWER_README.md",
        "data_statement.md",
    ]
    sums = {}
    for name in order:
        p = CONTENTS / name
        if not p.exists():
            raise SystemExit(f"missing bundle member: {p}")
        sums[name] = sha256(p)

    # Write checksums.sha256 (the authoritative sha256sum -c file).
    checksum_path = CONTENTS / "checksums.sha256"
    checksum_path.write_text(
        "".join(f"{sums[n]}  {n}\n" for n in order)
    )
    print(f"wrote {checksum_path.relative_to(ROOT)}")

    # Update MANIFEST.md (preserve header/footer; only refresh the code block).
    manifest_path = CONTENTS / "MANIFEST.md"
    header = "# Submission bundle MANIFEST — Tinker RL Lab (NeurIPS 2026 D&B)\n\n"
    header += (
        "Every file in `neurips2026_tinker_rl_lab.zip` with SHA-256. The "
        "machine-readable\nchecksum list lives in the companion file "
        "`checksums.sha256`, which is the\nauthoritative input to "
        "`sha256sum -c`.\n\n```\n"
    )
    body = "".join(f"{sums[n]}  {n}\n" for n in order)
    footer = (
        "```\n\nVerify with:\n\n```bash\nunzip neurips2026_tinker_rl_lab.zip\n"
        "sha256sum -c checksums.sha256\n```\n\n"
        f"Bundle composition: 7 files ({len(order)} data files + "
        "`checksums.sha256` / `MANIFEST.md`).\n"
    )
    manifest_path.write_text(header + body + footer)
    print(f"wrote {manifest_path.relative_to(ROOT)}")

    # Build the zip deterministically (sorted names, no absolute paths).
    ZIP_PATH.parent.mkdir(parents=True, exist_ok=True)
    members = sorted(
        [CONTENTS / n for n in order]
        + [checksum_path, manifest_path]
    )
    with zipfile.ZipFile(
        ZIP_PATH, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as zf:
        for m in members:
            zf.write(m, arcname=m.name)
    size_mb = ZIP_PATH.stat().st_size / (1024 * 1024)
    zip_sha = sha256(ZIP_PATH)
    print(
        f"wrote {ZIP_PATH.relative_to(ROOT)} ({size_mb:.1f} MB), "
        f"sha256={zip_sha}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
