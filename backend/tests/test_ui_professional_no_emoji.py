"""TDD guard for the professional, emoji-free UI and documentation policy.

Policy:
1. No emoji or pictographic symbols anywhere in application code, UI, docs, specs, IaC or tests.
   Typographic characters (arrows, section signs, bullets, block/box-drawing glyphs) remain allowed.
2. The backend answer marker used by Self-RAG / Judge ("Verified Inline Citations") is consistent and plain text.
3. The Gradio studio exposes a deliberate professional theme and stylesheet, and no rendered component label,
   value, or button text contains an emoji.
"""

import importlib.util
import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]

EMOJI_RE = re.compile(
    "["
    "\U0001F000-\U0001FAFF"  # pictographs, emoticons, transport, supplemental symbols
    "\u2600-\u27BF"          # misc symbols and dingbats (check marks, crosses, warning, lightning, gear)
    "\u2B00-\u2BFF"          # stars and large geometric symbols
    "\u23E9-\u23FF"          # timers, hourglass
    "\u25B6\u25C0"           # play / reverse play triangles
    "\uFE0F\u200D"           # variation selector and zero width joiner
    "]"
)

SCAN_DIRS = ["backend", "frontend", "docs", "specs", ".specify", "deploy", "observability", "scripts", ".github"]
SCAN_FILES = ["README.md", "docker-compose.yml"]
SCAN_SUFFIXES = {".py", ".md", ".yml", ".yaml", ".tf", ".json", ".txt"}
SKIP_PARTS = {".venv", "__pycache__", ".pytest_cache", "graphify-out", "node_modules"}


def _iter_files():
    for name in SCAN_FILES:
        p = REPO_ROOT / name
        if p.exists():
            yield p
    for d in SCAN_DIRS:
        base = REPO_ROOT / d
        if not base.exists():
            continue
        for p in base.rglob("*"):
            if p.is_file() and p.suffix.lower() in SCAN_SUFFIXES and not (set(p.parts) & SKIP_PARTS):
                yield p


def test_no_emoji_anywhere_in_application_docs_or_tests():
    offenders = []
    for path in _iter_files():
        text = path.read_text(encoding="utf-8", errors="ignore")
        for lineno, line in enumerate(text.splitlines(), start=1):
            found = EMOJI_RE.findall(line)
            if found:
                offenders.append(f"{path.relative_to(REPO_ROOT)}:{lineno} -> {''.join(sorted(set(found)))}")
    assert not offenders, "Emoji found (policy: professional plain-text UI):\n" + "\n".join(offenders[:40])


def test_citation_marker_is_plain_text_and_consistent_between_nodes_and_judge():
    nodes_src = (REPO_ROOT / "backend" / "app" / "agent" / "nodes.py").read_text(encoding="utf-8")
    judge_src = (REPO_ROOT / "backend" / "app" / "evaluation" / "judge.py").read_text(encoding="utf-8")
    assert "### Verified Inline Citations & Evidence Quotes" in nodes_src
    assert '"### Verified Inline Citations"' in judge_src


def _load_frontend_module():
    spec = importlib.util.spec_from_file_location("studio_frontend_app", REPO_ROOT / "frontend" / "app.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_gradio_studio_has_professional_theme_and_emoji_free_components():
    src = (REPO_ROOT / "frontend" / "app.py").read_text(encoding="utf-8")
    assert "gr.themes." in src, "Studio must define an explicit professional Gradio theme"
    assert "STUDIO_CSS" in src and "css=STUDIO_CSS" in src, "Studio must ship a dedicated stylesheet"
    assert "theme=STUDIO_THEME" in src, "Studio must launch with its professional theme"

    module = _load_frontend_module()
    offenders = []
    for block in module.demo.blocks.values():
        for attr in ("label", "value", "title"):
            val = getattr(block, attr, None)
            if isinstance(val, str) and EMOJI_RE.search(val):
                offenders.append(f"{type(block).__name__}.{attr}={val[:60]!r}")
    assert not offenders, "Emoji in rendered Gradio components: " + "; ".join(offenders[:20])
