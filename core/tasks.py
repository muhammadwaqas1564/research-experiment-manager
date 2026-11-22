"""Deterministic task-file creation (Markdown work orders for researchers)."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from core.paths import safe_filename, safe_resolve


def write_task_file(tasks_dir: Path, exp_id: str, title: str, body: str, params: dict[str, Any] | None = None) -> Path:
    tasks_dir.mkdir(parents=True, exist_ok=True)
    filename = safe_filename(f"{exp_id}-task.md")
    path = safe_resolve(tasks_dir, filename)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        f"# {exp_id}: {title}",
        "",
        f"_Created {stamp} by the research experiment manager._",
        "",
        "## Instructions",
        body.strip(),
        "",
    ]
    if params:
        lines += ["## Parameters", "```json", json.dumps(params, indent=2, sort_keys=True), "```", ""]
    lines += [
        "## When finished",
        f"- Save predictions as `results/{exp_id}_predictions.csv` (columns: y_true,y_pred)",
        f"- Save summary metrics as `results/{exp_id}_results.csv` (columns: metric,value)",
        "- Mark the experiment as completed in the registry",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
