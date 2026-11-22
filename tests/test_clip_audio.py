"""Exercise long-recording conversion, exact cuts and non-destructive exports."""
from fractions import Fraction
import math
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import wave

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from clip_audio import clip_filename, default_clip_name, export_segments, prepare_audio, validate_segment
from data import SAMPLE_RATE
from review_store import import_recording, load_project, save_project


class ClipAudioTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def wav(self, name="원본 녹음.wav", seconds=2):
        frames = round(seconds * SAMPLE_RATE)
        samples = np.random.default_rng(11).integers(-10000, 10001, frames, dtype=np.int16)
        path = self.root / name
        with wave.open(str(path), "wb") as output:
            output.setnchannels(1)
            output.setsampwidth(2)
            output.setframerate(SAMPLE_RATE)
            output.writeframes(samples.astype("<i2", copy=False).tobytes())
        return path, samples

    def prepared(self, source):
        audio = prepare_audio(source)
        self.addCleanup(audio.close)
        return audio

    def read_wav(self, path):
        with wave.open(str(path), "rb") as recording:
            self.assertEqual((recording.getframerate(), recording.getnchannels(),
                              recording.getsampwidth()), (SAMPLE_RATE, 1, 2))
            frames = recording.getnframes()
            pcm = recording.readframes(frames)
            self.assertEqual(len(pcm), frames * 2)
        return np.frombuffer(pcm, dtype="<i2")

    def test_long_recording_exact_cuts_and_reimport_keep_source_and_shared_session(self):
        source, samples = self.wav(seconds=90.03125)
        original = source.read_bytes()
        audio = self.prepared(source)
        self.assertEqual(audio.source, source.resolve())
        self.assertNotEqual(audio.path, source)
        self.assertEqual(audio.frames, len(samples))
        self.assertEqual(audio.duration, len(samples) / SAMPLE_RATE)
        self.assertEqual(len(audio.peaks), math.ceil(len(samples) / 1600))
        self.assertTrue(all(0 <= peak <= 1 for peak in audio.peaks))
        self.assertAlmostEqual(audio.peaks[-1], np.abs(samples[-500:].astype("int32")).max() / 32768)

        ranges = [(1.23456, 3.12345), (60, 90)]
        paths = export_segments(audio, ranges, self.root / "편집 결과")
        self.assertEqual(len(paths), 2)
        for path, (start, end) in zip(paths, ranges):
            first, last = validate_segment(audio, start, end)
            np.testing.assert_array_equal(self.read_wav(path), samples[first:last])

        reopened = self.prepared(source)
        renamed = self.root / "다른 이름.wav"
        renamed.write_bytes(original)
        copied = self.prepared(renamed)
        self.assertEqual(audio.session, reopened.session)
        self.assertEqual(audio.session, copied.session)
        self.assertTrue(audio.session.strip())
        project = self.root / "학습 자료"
        records = [import_recording(path, project, audio.session) for path in paths]
        self.assertEqual({row["session"] for row in records}, {audio.session})
        self.assertTrue(all(row["text"] == "" and row["reviewed"] is False for row in records))
        manifest, count = save_project(project, records)
        self.assertEqual(count, 0)
        self.assertEqual(manifest.read_bytes(), b"")
        self.assertEqual(load_project(project), records)
        self.assertEqual(source.read_bytes(), original)
        temporary = audio.path
        audio.close()
        audio.close()
        self.assertFalse(temporary.exists())
        self.assertEqual(source.read_bytes(), original)

    def test_long_phone_m4a_resampling_keeps_the_end_of_the_recording(self):
        import av
        if "aac" not in av.codecs_available:
            self.skipTest("AAC encoder is unavailable")
        source = self.root / "휴대폰 긴 녹음.m4a"
        rate = 44100
        frames = round(60.037 * rate)
        with av.open(str(source), "w") as container:
            stream = container.add_stream("aac", rate=rate)
            stream.layout = "stereo"
            for start in range(0, frames, 4096):
                positions = np.arange(start, min(start + 4096, frames))
                values = (.15 * np.sin(positions * 2 * np.pi * 430 / rate)).astype("float32")
                frame = av.AudioFrame.from_ndarray(np.vstack((values, values * .5)),
                                                    format="fltp", layout="stereo")
                frame.sample_rate = rate
                frame.time_base = Fraction(1, rate)
                frame.pts = start
                for packet in stream.encode(frame):
                    container.mux(packet)
            for packet in stream.encode(None):
                container.mux(packet)
        original = source.read_bytes()
        with av.open(str(source)) as container:
            decoded_frames = sum(frame.samples for frame in container.decode(audio=0))
        audio = self.prepared(source)
        self.assertAlmostEqual(audio.frames, decoded_frames * SAMPLE_RATE / rate, delta=1)
        self.assertGreater(audio.duration, 59.9)
        self.assertLess(audio.duration, 60.2)
        [tail] = export_segments(audio, [(audio.duration - .3, audio.duration)],
                                 self.root / "phone-clips")
        self.assertEqual(len(self.read_wav(tail)), round(.3 * SAMPLE_RATE))
        self.assertGreater(np.abs(self.read_wav(tail).astype("int32")).max(), 1000)
        self.assertEqual(source.read_bytes(), original)

    def test_segment_validation_enforces_finite_nonempty_in_bounds_thirty_second_limit(self):
        source, _ = self.wav(seconds=33)
        audio = self.prepared(source)
        self.assertEqual(validate_segment(audio, 0, 30), (0, 30 * SAMPLE_RATE))
        self.assertEqual(validate_segment(audio, 1, 31), (SAMPLE_RATE, 31 * SAMPLE_RATE))
        self.assertEqual(validate_segment(audio, 2.0001875, 32.0001875), (32003, 512003))
        self.assertEqual(validate_segment(audio, .10003, .20002), (1600, 3200))
        for start, end in (
                (-1, 1), (0, 0), (2, 1), (0, 34), (0, 30 + 1 / SAMPLE_RATE),
                (0, 1e-9), (float("nan"), 1), (0, float("nan")),
                (float("inf"), 1), (0, float("inf")), (True, 2), (0, False),
                ("not a time", 1), (None, 1)):
            with self.subTest(start=start, end=end), self.assertRaises(ValueError):
                validate_segment(audio, start, end)

    def test_invalid_or_duplicate_batch_is_rejected_before_any_export(self):
        source, _ = self.wav()
        original = source.read_bytes()
        audio = self.prepared(source)
        for index, ranges in enumerate((
                [], [(0, 1), (1, 3)], [(0, 1), (0, 1)],
                [(0, 1), (1e-9, 1 + 1e-9)], [(0, 1), (1,)])):
            with self.subTest(ranges=ranges):
                destination = self.root / f"invalid-{index}"
                with self.assertRaises(ValueError):
                    export_segments(audio, ranges, destination)
                self.assertFalse(destination.exists())
        self.assertEqual(source.read_bytes(), original)

    def test_name_collisions_and_failed_batch_never_replace_existing_files(self):
        source, _ = self.wav()
        original = source.read_bytes()
        audio = self.prepared(source)
        destination = self.root / "기존 결과"
        [first] = export_segments(audio, [(0, .5)], destination)
        first.write_bytes(b"existing output must not be overwritten")
        [second] = export_segments(audio, [(0, .5)], destination)
        self.assertNotEqual(first, second)
        self.assertEqual(first.read_bytes(), b"existing output must not be overwritten")
        self.assertEqual(len(self.read_wav(second)), SAMPLE_RATE // 2)
        before = {path: path.read_bytes() for path in destination.iterdir()}
        original_open = Path.open
        attempts = []

        def fail_second_output(path, mode="r", *args, **kwargs):
            if mode == "xb":
                attempts.append(path)
                if len(attempts) == 2:
                    raise OSError("simulated output failure")
            return original_open(path, mode, *args, **kwargs)

        with patch("pathlib.Path.open", new=fail_second_output):
            with self.assertRaisesRegex(OSError, "simulated output failure"):
                export_segments(audio, [(0, .5), (.5, 1)], destination)
        self.assertEqual(len(attempts), 2)
        self.assertEqual({path: path.read_bytes() for path in destination.iterdir()}, before)
        self.assertEqual(source.read_bytes(), original)

    def test_invalid_sources_and_successful_close_remove_only_temporary_files(self):
        import soundfile as sf
        empty, _ = self.wav("empty.wav", seconds=0)
        corrupt = self.root / "corrupt.wav"
        corrupt.write_bytes(b"not an audio file")
        nonfinite = self.root / "nonfinite.wav"
        sf.write(nonfinite, np.array([0., np.nan, np.inf], dtype="float32"),
                 SAMPLE_RATE, subtype="FLOAT")
        sources = [empty, corrupt, nonfinite]
        originals = {path: path.read_bytes() for path in sources}
        temporary_root = self.root / "temporary"
        temporary_root.mkdir()
        with patch("tempfile.tempdir", str(temporary_root)):
            for source in [*sources, self.root / "missing.wav"]:
                with self.subTest(source=source.name), self.assertRaises((ValueError, OSError)):
                    prepare_audio(source)
                self.assertEqual(list(temporary_root.iterdir()), [])
            source, _ = self.wav("valid.wav")
            audio = prepare_audio(source)
            self.assertTrue(audio.path.exists())
            audio.close()
            self.assertEqual(list(temporary_root.iterdir()), [])
        for path, content in originals.items():
            self.assertEqual(path.read_bytes(), content)

    def test_corrupt_temporary_audio_does_not_leave_partial_exports(self):
        source, _ = self.wav()
        original = source.read_bytes()
        audio = self.prepared(source)
        content = audio.path.read_bytes()
        audio.path.write_bytes(content[:-10])
        destination = self.root / "truncated-output"
        with self.assertRaises(ValueError):
            export_segments(audio, [(0, audio.duration)], destination)
        self.assertFalse(list(destination.glob("*")))
        self.assertEqual(source.read_bytes(), original)

    def test_custom_names_keep_clip_contents_and_preserve_existing_casefold_names(self):
        source, samples = self.wav()
        original = source.read_bytes()
        audio = self.prepared(source)
        self.assertEqual(clip_filename("질문"), "질문.wav")
        self.assertEqual(clip_filename("Answer.WAV"), "Answer.WAV")
        self.assertEqual(default_clip_name(source, 1), source.stem + "_001.wav")
        destination = self.root / "custom"
        destination.mkdir()
        (destination / "질문.wav").write_bytes(b"existing question")
        (destination / "answer.wav").write_bytes(b"existing answer")
        paths = export_segments(audio, [(0, .5), (.5, 1)], destination,
                                names=["질문", "Answer.WAV"])
        self.assertEqual([path.name for path in paths], ["질문_2.wav", "Answer_2.WAV"])
        for path, (first, last) in zip(paths, ((0, 8000), (8000, 16000))):
            np.testing.assert_array_equal(self.read_wav(path), samples[first:last])
        self.assertEqual((destination / "질문.wav").read_bytes(), b"existing question")
        self.assertEqual((destination / "answer.wav").read_bytes(), b"existing answer")
        self.assertEqual(source.read_bytes(), original)

    def test_invalid_custom_names_and_batch_names_are_rejected_before_writes(self):
        source, _ = self.wav()
        audio = self.prepared(source)
        invalid = ("", " ", None, 7, ".wav", "../escape", "folder/name", "folder\\name",
                   "C:escape", "bad?name", "bad|name", "bad\nname", "tail.", "tail ",
                   "CON", "nul.wav", "COM1.txt", "LPT²", "CONIN$.wav", "x" * 241,
                   "가" * 80)
        for index, name in enumerate(invalid):
            with self.subTest(name=name):
                directory = self.root / f"bad-name-{index}"
                with self.assertRaises(ValueError):
                    export_segments(audio, [(0, .5)], directory, names=[name])
                self.assertFalse(directory.exists())
        for index, names in enumerate(([], ["one"], ["same", "SAME.WAV"], "not a list")):
            with self.subTest(names=names):
                directory = self.root / f"bad-batch-{index}"
                with self.assertRaises(ValueError):
                    export_segments(audio, [(0, .5), (.5, 1)], directory, names=names)
                self.assertFalse(directory.exists())
        self.assertEqual(clip_filename(default_clip_name(Path("가" * 100 + ".wav"), 1)),
                         default_clip_name(Path("가" * 100 + ".wav"), 1))

    def test_custom_name_failure_rolls_back_only_new_files(self):
        source, _ = self.wav()
        original = source.read_bytes()
        audio = self.prepared(source)
        destination = self.root / "rollback-custom"
        destination.mkdir()
        existing = destination / "clip.wav"
        existing.write_bytes(b"keep existing output")
        original_open = Path.open
        attempts = []

        def fail_second_output(path, mode="r", *args, **kwargs):
            if mode == "xb":
                attempts.append(path)
                if len(attempts) == 2:
                    raise OSError("custom output failure")
            return original_open(path, mode, *args, **kwargs)

        with patch("pathlib.Path.open", new=fail_second_output):
            with self.assertRaisesRegex(OSError, "custom output failure"):
                export_segments(audio, [(0, .5), (.5, 1)], destination, names=["clip", "second"])
        self.assertEqual(list(destination.iterdir()), [existing])
        self.assertEqual(existing.read_bytes(), b"keep existing output")
        self.assertEqual(source.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
