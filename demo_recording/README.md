# Demo deliverables

## Live URL

**https://arvindcr4-tinkerrl-bench-demo.hf.space/**

(Or via the Space page: https://huggingface.co/spaces/arvindcr4/tinkerrl-bench-demo)

The Space:
- 4 tabs: Tool-use comparison · ZVF diagnostic · Team artifacts · Headline numbers
- All outputs are pre-computed from the released model cards (no live inference, no GPU)
- Free-tier CPU-only — sleeps after 48 h idle, cold-starts in ~30 s
- Hit the URL once around 8 AM tomorrow to wake it before the talk

## Pre-recorded video

`demo.mp4` — 36 s, 714 KB, 1600×1000 h264 @ 30 fps. Drop into the deck at slide 8 (between "Tiered Protocol" and "Variance Methods") OR play it during Q&A as evidence of released artifacts.

Also includes individual screenshots: `01_tool_use.png`, `01b_metrics.png`, `02_zvf.png`, `03_artifacts.png`, `04_headline.png` — embed-ready.

## How to play during the talk

**Option A — Embed the MP4 in the .pptx (recommended).**
- In PowerPoint/Keynote: Insert → Video → "From File" — pick `demo.mp4`
- Set "Start: Automatically" on click; "Play full screen"; "Loop until stopped" off
- Test once on the projector machine before the talk

**Option B — Live URL.**
- Have the Space URL bookmarked on the presentation laptop
- Backup tab on a teammate's phone in case the projector chrome misbehaves
- Wake the Space at 8 AM tomorrow with one HTTP GET so it's hot when you click

**Option C — Static screenshots only.**
- If video playback is finicky on the projector machine, drop the four PNGs as a 4-slide mini-section in the deck
- Loses the "interactive" feel but cannot break

## Where in the talk to use it

Slide 8 (Tiered Protocol) → **demo interlude (90 s)** → Slide 9 (Variance methods).

Narration script for the interlude (have **Sandhya** lead — it's her work):

> "Before P4 walks through the variance-mitigation methods, here's what one of these tool-use experiments actually looks like. [Click play / open URL.]
>
> Left side — SFT-only output on the prompt 'what's the weather in Paris.' The model answers in plain text. Right side — the same model after SFT plus GRPO on the same prompt — emits a structured JSON tool call. Across twelve test queries, JSON validity goes from zero percent to ninety-two percent. That's the 'zero to ninety-two' line in the abstract.
>
> The other tabs hold the ZVF diagnostic and the team's released artifacts — all six members have their own HuggingFace or GitHub link. Now P4..."

## Recording the demo yourself (optional, if you want a longer cut)

The Space is the same; you can produce a longer narrated screencast using OBS Studio (already installed via Flatpak):

```bash
flatpak run com.obsproject.Studio
```

In OBS: add **Window Capture** for the Brave/Chromium window, set output to MP4 1080p 30 fps, click "Start Recording", click through the tabs while narrating into the mic.

## Re-deploying after rotating the API token

After tomorrow's viva, rotate the HF token at https://huggingface.co/settings/tokens and re-deploy:

```bash
export HF_TOKEN=<new_token>
python3 -c "
from huggingface_hub import HfApi
HfApi(token='$HF_TOKEN').upload_folder(
  folder_path='/tmp/demo_space',
  repo_id='arvindcr4/tinkerrl-bench-demo',
  repo_type='space',
)
"
```

The Space source is in `/tmp/demo_space/` — `app.py`, `requirements.txt`, `README.md`. If `/tmp` got cleared, all three files are reproducible from this conversation.

## Pre-talk wake-up command (run at ~8 AM tomorrow)

```bash
curl -s -o /dev/null -w "%{http_code}\n" https://arvindcr4-tinkerrl-bench-demo.hf.space/
```

If it returns 200 immediately, the Space is hot. If it takes 20-30 s, the cold-start is happening — the Space will be ready a minute later.
