"""Small process helpers for the review app; callers run preparation off the UI thread."""
from contextlib import suppress
from datetime import datetime, timezone
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import uuid


def _unique_name(prefix=""):
    return prefix + datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex


def training_dependencies():
    return [name for name in ("torch", "transformers", "peft")
            if importlib.util.find_spec(name) is None]


def prepare_project(project_dir, previous_data=None):
    from prepare import prepare_dataset

    project = Path(project_dir).expanduser().resolve()
    manifest = project / "manifest.jsonl"
    if not manifest.is_file():
        raise FileNotFoundError("저장된 학습 자료가 없습니다. 먼저 정답을 확인하고 일괄 저장하세요.")
    output = project / "prepared" / _unique_name()
    previous = Path(previous_data).expanduser().resolve() if previous_data else None
    prepare_dataset(manifest, output, previous_data=previous)
    return output


def start_training(project_dir, prepared_path, device="auto", adapter=None, quick=False):
    if device not in ("auto", "cpu", "cuda"):
        raise ValueError("학습 장치는 auto, cpu 또는 cuda를 선택하세요.")
    project = Path(project_dir).expanduser().resolve()
    prepared = Path(prepared_path).expanduser().resolve()
    for name in ("train", "validation", "test"):
        if not (prepared / f"{name}.jsonl").is_file():
            raise FileNotFoundError(f"학습 자료 준비가 필요합니다: {prepared / (name + '.jsonl')}")
    command = [sys.executable, "-u", str(Path(__file__).resolve().with_name("train.py")),
               "--data", str(prepared), "--device", device]
    if adapter:
        adapter = Path(adapter).expanduser().resolve()
        if not adapter.is_dir() or any(not (adapter / name).is_file() for name in (
            "adapter_config.json", "training_info.json",
        )):
            raise ValueError("기존 학습 결과의 best 폴더를 선택하세요. 어댑터 설정과 학습 기록이 필요합니다.")
        command += ["--adapter", str(adapter)]
    output = project / "runs" / _unique_name("voice-")
    output.parent.mkdir(parents=True, exist_ok=True)
    command += ["--output", str(output)]
    if quick:
        command += ["--epochs", "1", "--max-steps", "1"]
    environment = os.environ.copy()
    environment["PYTHONIOENCODING"] = "utf-8"
    process = subprocess.Popen(
        command, cwd=Path(__file__).resolve().parent, env=environment,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        encoding="utf-8", errors="replace", shell=False,
    )
    return process, output


def stop_training(process):
    """Stop one owned training process; call off the UI thread (up to six seconds)."""
    stopped = False
    try:
        if process.poll() is None:
            with suppress(ProcessLookupError):
                process.terminate()
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                with suppress(ProcessLookupError):
                    process.kill()
                process.wait(timeout=3)
        stopped = True
    finally:
        if stopped and process.stdout is not None:
            process.stdout.close()
