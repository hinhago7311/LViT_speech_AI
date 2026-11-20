"""Reviewed audio/text manifests for Whisper training."""
import hashlib
import json
import math
from pathlib import Path
import random

SAMPLE_RATE = 16000
MAX_SECONDS = 30


def read_manifest(path):
    """Read reviewed JSONL rows; audio paths are relative to the manifest."""
    path = Path(path).expanduser().resolve()
    rows = []
    with path.open(encoding="utf-8-sig") as source:
        for number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError("각 줄은 JSON 객체여야 합니다.")
                for key in ("audio", "text", "session"):
                    if not isinstance(row.get(key), str) or not row[key].strip():
                        raise ValueError(f"{key}에 비어 있지 않은 문자열이 필요합니다.")
                if row.get("reviewed") is not True:
                    raise ValueError("정답을 확인한 뒤 reviewed를 true로 설정하세요.")
                if row.get("language") not in ("ko", "en"):
                    raise ValueError("language는 ko 또는 en이어야 합니다.")
                audio = Path(row["audio"]).expanduser()
                if not audio.is_absolute():
                    audio = path.parent / audio
                audio = audio.resolve()
                if not audio.is_file():
                    raise ValueError(f"음성 파일이 없습니다: {audio}")
                rows.append({**row, "audio": str(audio), "text": row["text"].strip(),
                             "session": row["session"].strip()})
            except (ValueError, OSError) as error:
                raise ValueError(f"{path.name}:{number}: {error}") from error
    if not rows:
        raise ValueError("학습 자료가 없습니다. 확인한 음성과 정답을 추가하세요.")
    return rows


def load_audio(path):
    """Decode to finite float32 mono at 16 kHz, rejecting overlong audio."""
    import numpy as np
    import soundfile as sf
    from scipy.signal import resample_poly

    try:
        info = sf.info(str(path))
        if info.frames <= 0 or info.samplerate <= 0 or info.channels <= 0:
            raise ValueError("비어 있거나 잘못된 음성 파일입니다.")
        if info.frames / info.samplerate > MAX_SECONDS:
            raise ValueError("30초를 초과합니다. 발화별로 나눠 정답을 연결하세요.")
        samples, rate = sf.read(str(path), dtype="float32", always_2d=True)
    except (RuntimeError, OSError) as error:
        raise ValueError(f"음성을 읽을 수 없습니다: {path}: {error}") from error
    if not np.isfinite(samples).all():
        raise ValueError(f"음성에 NaN/무한대 값이 있습니다: {path}")
    samples = samples.mean(axis=1)
    if rate != SAMPLE_RATE:
        divisor = math.gcd(rate, SAMPLE_RATE)
        samples = resample_poly(samples, SAMPLE_RATE // divisor, rate // divisor)
    samples = np.ascontiguousarray(samples, dtype="float32")
    if not samples.size or not np.isfinite(samples).all():
        raise ValueError(f"변환한 음성이 비어 있거나 잘못되었습니다: {path}")
    return samples


def validate_audio(path):
    """Return duration and fingerprint of decoded, mono/16 kHz samples."""
    samples = load_audio(path)
    fingerprint = hashlib.sha256(samples.astype("<f4", copy=False).tobytes()).hexdigest()
    return len(samples) / SAMPLE_RATE, fingerprint


def split_by_session(rows, seed=42, previous_assignment=None):
    """Split sessions; preserve previous assignments when adding training data."""
    sessions = set()
    for row in rows:
        session = row.get("session")
        if not isinstance(session, str) or not session.strip():
            raise ValueError("모든 자료에 녹음 세션 ID가 필요합니다.")
        sessions.add(session)
    if previous_assignment is None:
        if len(sessions) < 3:
            raise ValueError("train/validation/test 분리를 위해 서로 다른 녹음 세션 3개 이상이 필요합니다.")
        sessions = sorted(sessions)
        random.Random(seed).shuffle(sessions)
        count = max(1, round(len(sessions) * .1))
        assignment = {session: "validation" for session in sessions[:count]}
        assignment.update({session: "test" for session in sessions[count:2 * count]})
    else:
        assignment = previous_assignment
        if any(name not in ("train", "validation", "test") for name in assignment.values()):
            raise ValueError("이전 세션의 분할 이름은 train/validation/test여야 합니다.")
    result = {"train": [], "validation": [], "test": []}
    for row in rows:
        result[assignment.get(row["session"], "train")].append(row)
    if not all(result.values()):
        raise ValueError("세 분할에 자료가 필요합니다. 이전 validation/test 자료를 새 manifest에도 포함하세요.")
    return result
