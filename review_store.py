"""Local reviewed-recording projects; originals are never changed."""
import hashlib
from itertools import chain
import json
import os
from pathlib import Path
import tempfile
import wave

import numpy as np

from data import MAX_SECONDS, SAMPLE_RATE, load_audio, read_manifest


def _decode_recording(source):
    if source.suffix.lower() in (".wav", ".wave", ".flac", ".ogg"):
        return load_audio(source)
    try:
        import av
    except ImportError as error:
        raise ValueError("M4A/MP3/AAC를 읽으려면 av 패키지를 설치하세요: pip install av") from error
    chunks, total = [], 0
    try:
        with av.open(str(source)) as recording:
            if not recording.streams.audio:
                raise ValueError("파일에 음성 트랙이 없습니다.")
            resampler = av.AudioResampler(format="fltp", layout="mono", rate=SAMPLE_RATE)
            for frame in chain(recording.decode(audio=0), [None]):
                for converted in resampler.resample(frame):
                    samples = converted.to_ndarray().reshape(-1).astype("float32", copy=False)
                    total += samples.size
                    if total > SAMPLE_RATE * MAX_SECONDS:
                        raise ValueError("30초를 초과합니다. 발화별로 나눠 정답을 연결하세요.")
                    if not np.isfinite(samples).all():
                        raise ValueError("음성에 NaN/무한대 값이 있습니다.")
                    chunks.append(samples)
    except av.FFmpegError as error:
        raise ValueError(f"음성을 읽을 수 없습니다: {source.name}: {error}") from error
    if not total:
        raise ValueError("비어 있는 음성 파일입니다.")
    return np.concatenate(chunks)


def _inside(path, directory):
    path = path.resolve()
    if not path.is_relative_to(directory):
        raise ValueError(f"프로젝트 밖의 파일은 사용할 수 없습니다: {path}")
    return path


def _canonical_info(path):
    try:
        with wave.open(str(path), "rb") as recording:
            frames = recording.getnframes()
            if (recording.getnchannels(), recording.getsampwidth(), recording.getframerate(),
                    recording.getcomptype()) != (1, 2, SAMPLE_RATE, "NONE"):
                raise ValueError("프로젝트 음성은 16kHz mono PCM16 WAV여야 합니다.")
            if not 0 < frames <= SAMPLE_RATE * MAX_SECONDS:
                raise ValueError("프로젝트 음성은 비어 있지 않은 30초 이하 녹음이어야 합니다.")
            pcm = recording.readframes(frames)
            if len(pcm) != frames * 2:
                raise ValueError("WAV 음성 데이터가 손상되었습니다.")
    except (wave.Error, EOFError) as error:
        raise ValueError(f"프로젝트 WAV 파일이 잘못되었습니다: {path.name}") from error
    fingerprint = hashlib.sha256(pcm).hexdigest()
    if path.name != f"{fingerprint}.wav":
        raise ValueError(f"프로젝트 음성의 이름과 내용이 일치하지 않습니다: {path.name}")
    return frames / SAMPLE_RATE, fingerprint


def import_recording(source, project_dir, session, language="ko"):
    source = Path(source).expanduser().resolve()
    if not source.is_file():
        raise ValueError(f"음성 파일이 없습니다: {source}")
    if not isinstance(session, str) or language not in ("ko", "en"):
        raise ValueError("세션은 문자열, language는 ko 또는 en이어야 합니다.")
    samples = _decode_recording(source)
    pcm = np.rint(np.clip(samples, -1, 32767 / 32768) * 32768).astype("<i2").tobytes()
    fingerprint = hashlib.sha256(pcm).hexdigest()
    project = Path(project_dir).expanduser().resolve()
    audio_dir = _inside(project / "audio", project)
    destination = _inside(audio_dir / f"{fingerprint}.wav", project)
    audio_dir.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        _canonical_info(destination)
    else:
        with destination.open("xb") as output:
            try:
                with wave.open(output, "wb") as recording:
                    recording.setnchannels(1)
                    recording.setsampwidth(2)
                    recording.setframerate(SAMPLE_RATE)
                    recording.writeframes(pcm)
            except Exception:
                destination.unlink(missing_ok=True)
                raise
    return {"id": fingerprint, "audio": destination.relative_to(project).as_posix(),
            "source_name": source.name, "text": "", "session": session.strip(),
            "language": language, "reviewed": False, "duration": len(pcm) / 2 / SAMPLE_RATE,
            "audio_sha256": fingerprint}


def _validated_row(row, project):
    if not isinstance(row, dict) or not isinstance(row.get("audio"), str):
        raise ValueError("행마다 음성 경로가 필요합니다.")
    audio = Path(row["audio"])
    if not audio.is_absolute():
        audio = project / audio
    audio = _inside(audio, project)
    if audio.parent != _inside(project / "audio", project):
        raise ValueError("프로젝트 audio 폴더의 음성만 저장할 수 있습니다.")
    duration, fingerprint = _canonical_info(audio)
    if type(row.get("reviewed")) is not bool:
        raise ValueError("reviewed는 true 또는 false여야 합니다.")
    if not isinstance(row.get("text"), str) or not isinstance(row.get("session"), str):
        raise ValueError("정답과 세션은 문자열이어야 합니다.")
    if row.get("language") not in ("ko", "en"):
        raise ValueError("language는 ko 또는 en이어야 합니다.")
    if row["reviewed"] and (not row["text"].strip() or not row["session"].strip()):
        raise ValueError("검수 완료한 행에는 정답과 녹음 세션이 필요합니다.")
    return {**row, "id": fingerprint, "audio": audio.relative_to(project).as_posix(),
            "duration": duration, "audio_sha256": fingerprint}


def load_project(project_dir):
    project = Path(project_dir).expanduser().resolve()
    draft, manifest = project / ".lvit-review.json", project / "manifest.jsonl"
    if draft.exists():
        try:
            rows = json.loads(draft.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise ValueError(f"프로젝트 임시 저장 JSON이 손상되었습니다: {error}") from error
        if not isinstance(rows, list):
            raise ValueError("프로젝트 임시 저장은 행 목록이어야 합니다.")
        result = [_validated_row(row, project) for row in rows]
    elif manifest.exists() and manifest.stat().st_size:
        result = []
        for row in read_manifest(manifest):
            source = _inside(Path(row["audio"]), project)
            imported = import_recording(source, project, row["session"], row["language"])
            result.append({**imported, "source_name": row.get("source_name", imported["source_name"]),
                           "text": row["text"], "reviewed": True})
    else:
        return []
    if len({row["id"] for row in result}) != len(result):
        raise ValueError("프로젝트에 중복된 음성 행이 있습니다.")
    return result


def _atomic_write(path, text):
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=f".{path.name}-", delete=False) as output:
            temporary = Path(output.name)
            output.write(text)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def save_project(project_dir, rows):
    project = Path(project_dir).expanduser().resolve()
    normalized = [_validated_row(row, project) for row in rows]
    if len({row["id"] for row in normalized}) != len(normalized):
        raise ValueError("중복 음성 행을 제거한 뒤 저장하세요.")
    reviewed = [row for row in normalized if row["reviewed"]]
    manifest = project / "manifest.jsonl"
    project.mkdir(parents=True, exist_ok=True)
    _atomic_write(project / ".lvit-review.json", json.dumps(normalized, ensure_ascii=False, indent=2))
    _atomic_write(manifest, "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in reviewed))
    return manifest, len(reviewed)
