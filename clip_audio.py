"""Stream long recordings to a temporary WAV and export reviewed clip ranges."""
from dataclasses import dataclass, field
import hashlib
from itertools import chain
import math
from pathlib import Path
import re
import tempfile
import wave

import numpy as np

from data import MAX_SECONDS, SAMPLE_RATE

PEAK_FRAMES = SAMPLE_RATE // 10
MAX_WAV_FRAMES = (0xFFFFFFFF - 36) // 2


@dataclass
class PreparedAudio:
    source: Path
    path: Path
    frames: int
    peaks: list[float]
    session: str
    _temporary: tempfile.TemporaryDirectory = field(repr=False)

    @property
    def duration(self):
        return self.frames / SAMPLE_RATE

    def close(self):
        """Delete this object's temporary copy, never its source recording."""
        self._temporary.cleanup()


def prepare_audio(source):
    """Decode one recording without retaining its complete audio in memory."""
    import av

    source = Path(source).expanduser().resolve()
    if not source.is_file():
        raise ValueError(f"원본 음성 파일이 없습니다: {source}")
    temporary = tempfile.TemporaryDirectory(prefix="lvit-audio-")
    path = Path(temporary.name) / "audio.wav"
    frames, peak_count, peak = 0, 0, 0.0
    peaks, digest = [], hashlib.sha256()
    try:
        with av.open(str(source)) as recording, wave.open(str(path), "wb") as output:
            if not recording.streams.audio:
                raise ValueError("파일에 음성 트랙이 없습니다.")
            output.setnchannels(1)
            output.setsampwidth(2)
            output.setframerate(SAMPLE_RATE)
            resampler = av.AudioResampler(format="fltp", layout="mono", rate=SAMPLE_RATE)
            # None flushes samples retained by the rate converter at EOF.
            for frame in chain(recording.decode(audio=0), [None]):
                for converted in resampler.resample(frame):
                    samples = converted.to_ndarray().reshape(-1)
                    if not np.isfinite(samples).all():
                        raise ValueError("원본 음성에 NaN 또는 무한대 값이 있습니다.")
                    pcm = np.rint(np.clip(samples, -1, 32767 / 32768) * 32768).astype("<i2")
                    if frames + pcm.size > MAX_WAV_FRAMES:
                        raise ValueError("변환한 음성이 WAV의 약 4GB 한도를 넘습니다. 원본을 나눠 주세요.")
                    content = pcm.tobytes()
                    output.writeframesraw(content)
                    digest.update(content)
                    frames += pcm.size
                    position = 0
                    while position < pcm.size:
                        count = min(PEAK_FRAMES - peak_count, pcm.size - position)
                        values = pcm[position:position + count].astype("int32")
                        peak = max(peak, float(np.abs(values).max()) / 32768)
                        position += count
                        peak_count += count
                        if peak_count == PEAK_FRAMES:
                            peaks.append(peak)
                            peak_count, peak = 0, 0.0
        if not frames:
            raise ValueError("원본 음성이 비어 있습니다.")
        if peak_count:
            peaks.append(peak)
        return PreparedAudio(source, path, frames, peaks, "source-" + digest.hexdigest()[:24], temporary)
    except Exception as error:
        temporary.cleanup()
        if isinstance(error, (av.FFmpegError, wave.Error, EOFError)):
            raise ValueError(f"원본 음성을 읽을 수 없습니다: {source.name}: {error}") from error
        raise


def validate_segment(audio, start, end):
    """Return sample bounds for a finite, nonempty clip of at most 30 seconds."""
    if isinstance(start, (bool, np.bool_)) or isinstance(end, (bool, np.bool_)):
        raise ValueError("시작과 끝은 초 단위 숫자로 입력하세요.")
    try:
        start, end = float(start), float(end)
    except (TypeError, ValueError) as error:
        raise ValueError("시작과 끝은 초 단위 숫자로 입력하세요.") from error
    if not math.isfinite(start) or not math.isfinite(end):
        raise ValueError("시작과 끝은 유한한 숫자여야 합니다.")
    if not 0 <= start < end <= audio.duration:
        raise ValueError(f"구간은 0초 이상, 시작보다 끝이 크고 원본 길이 {audio.duration:.6f}초 이하여야 합니다.")
    first, last = round(start * SAMPLE_RATE), round(end * SAMPLE_RATE)
    if first >= last:
        raise ValueError("선택한 구간에 실제 음성 샘플이 없습니다.")
    if last - first > MAX_SECONDS * SAMPLE_RATE:
        raise ValueError("한 구간은 30초 이하여야 합니다. 여러 구간으로 나눠 주세요.")
    return first, last


def _filename_prefix(source):
    prefix = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", source.stem).strip(" .")
    return prefix[:80].rstrip(" .") or "audio"


def clip_filename(name):
    """Return a safe user-selected WAV filename without changing its basename."""
    if not isinstance(name, str) or not name.strip():
        raise ValueError("각 구간에 비어 있지 않은 파일명을 입력하세요.")
    if name.endswith(".") or name[-1].isspace() or re.search(r'[<>:"/\\|?*\x00-\x1f]', name):
        raise ValueError("파일명에 경로·금지 문자를 넣거나 끝에 공백·점을 붙일 수 없습니다.")
    filename = name if name.lower().endswith(".wav") else name + ".wav"
    if not filename[:-4].strip(" ."):
        raise ValueError("확장자 앞에 파일명을 입력하세요.")
    device = filename.split(".", 1)[0].rstrip(" ").upper()
    if device in {"CON", "PRN", "AUX", "NUL", "CONIN$", "CONOUT$"} or re.fullmatch(r"(?:COM|LPT)[1-9¹²³]", device):
        raise ValueError("Windows 예약 장치명은 파일명으로 사용할 수 없습니다.")
    if len(filename.encode("utf-8")) > 240 or len(filename.encode("utf-16-le")) // 2 > 240:
        raise ValueError("파일명이 너무 깁니다. 더 짧은 이름을 입력하세요.")
    return filename


def default_clip_name(source, number):
    """Return the original safe prefix_001.wav naming pattern."""
    if isinstance(number, bool) or not isinstance(number, int) or number < 1:
        raise ValueError("파일 번호는 1 이상의 정수여야 합니다.")
    prefix = _filename_prefix(Path(source))
    suffix = f"_{number:03d}.wav"
    while prefix and len((prefix + suffix).encode("utf-8")) > 240:
        prefix = prefix[:-1]
    return clip_filename((prefix or "audio") + suffix)


def export_segments(audio, ranges, output_dir, names=None):
    """Write new WAV files exclusively; roll back only files made by this call."""
    segments = []
    for selected in ranges:
        if not isinstance(selected, (tuple, list)) or len(selected) != 2:
            raise ValueError("각 구간은 (시작 초, 끝 초) 두 숫자로 지정하세요.")
        segments.append(validate_segment(audio, *selected))
    if not segments:
        raise ValueError("저장할 음성 구간을 하나 이상 선택하세요.")
    if len(set(segments)) != len(segments):
        raise ValueError("동일한 음성 구간이 중복되어 있습니다.")
    filenames = None
    if names is not None:
        if isinstance(names, (str, bytes)):
            raise ValueError("구간별 파일명을 목록으로 지정하세요.")
        try:
            filenames = [clip_filename(name) for name in names]
        except TypeError as error:
            raise ValueError("구간별 파일명을 목록으로 지정하세요.") from error
        if len(filenames) != len(segments):
            raise ValueError("선택한 구간 수와 파일명 수가 같아야 합니다.")
        if len({name.casefold() for name in filenames}) != len(filenames):
            raise ValueError("구간별 파일명이 중복되어 있습니다. 대소문자만 다른 이름도 사용할 수 없습니다.")
    reserved = {name.casefold() for name in filenames or []}
    directory = Path(output_dir).expanduser().resolve()
    created, number = [], 1
    try:
        with wave.open(str(audio.path), "rb") as source:
            if (source.getnchannels(), source.getsampwidth(), source.getframerate(),
                    source.getcomptype(), source.getnframes()) != (1, 2, SAMPLE_RATE, "NONE", audio.frames):
                raise ValueError("임시 음성 파일이 손상되었거나 준비한 형식과 다릅니다.")
            directory.mkdir(parents=True, exist_ok=True)
            occupied = {path.name.casefold() for path in directory.iterdir()}
            for index, (first, last) in enumerate(segments):
                suffix = 1
                while True:
                    if filenames is None:
                        filename = default_clip_name(audio.source, number)
                        number += 1
                    else:
                        requested = filenames[index]
                        filename = requested if suffix == 1 else f"{requested[:-4]}_{suffix}{requested[-4:]}"
                        suffix += 1
                        if filename.casefold() != requested.casefold() and filename.casefold() in reserved:
                            continue
                    if filename.casefold() in occupied:
                        continue
                    if len(filename.encode("utf-8")) > 255 or len(filename.encode("utf-16-le")) // 2 > 255:
                        raise ValueError("충돌 번호를 붙일 수 있도록 파일명을 더 짧게 입력하세요.")
                    destination = directory / filename
                    try:
                        handle = destination.open("xb")
                    except FileExistsError:
                        occupied.add(filename.casefold())
                        continue
                    occupied.add(filename.casefold())
                    break
                created.append(destination)
                with handle, wave.open(handle, "wb") as output:
                    output.setnchannels(1)
                    output.setsampwidth(2)
                    output.setframerate(SAMPLE_RATE)
                    source.setpos(first)
                    remaining = last - first
                    while remaining:
                        count = min(remaining, SAMPLE_RATE)
                        content = source.readframes(count)
                        if len(content) != count * 2:
                            raise ValueError("임시 음성 데이터가 손상되어 구간을 읽을 수 없습니다.")
                        output.writeframesraw(content)
                        remaining -= count
        return created
    except Exception:
        for destination in created:
            destination.unlink(missing_ok=True)
        raise
