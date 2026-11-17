import json
from pathlib import Path
import sys
import tempfile
import unittest
import wave

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data import load_audio, read_manifest, split_by_session, validate_audio
from prepare import prepare_dataset


class DataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.addCleanup(self.temp.cleanup)

    def wav(self, name, frequency=220, rate=16000, seconds=.1, channels=1):
        values = np.sin(np.arange(int(rate * seconds)) * 2 * np.pi * frequency / rate)
        values = (values * 4000).astype("<i2")
        if channels == 2:
            values = np.column_stack((values, values))
        path = self.root / name
        with wave.open(str(path), "wb") as stream:
            stream.setnchannels(channels)
            stream.setsampwidth(2)
            stream.setframerate(rate)
            stream.writeframes(values.tobytes())
        return path

    def manifest(self, rows):
        path = self.root / "source.jsonl"
        path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows), encoding="utf-8")
        return path

    def row(self, audio, session="day1", **changes):
        return {"audio": audio, "text": "지금 마이크 소리 잘 들리시나요?",
                "session": session, "language": "ko", "reviewed": True, **changes}

    def test_manifest_requires_reviewed_correction_and_resolves_relative_audio(self):
        self.wav("one.wav")
        row = read_manifest(self.manifest([self.row("one.wav")]))[0]
        self.assertEqual(row["audio"], str(self.root / "one.wav"))
        for change in ({"reviewed": False}, {"reviewed": 1}, {"text": " "},
                       {"language": "ja"}, {"session": ""}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                read_manifest(self.manifest([self.row("one.wav", **change)]))

    def test_split_is_deterministic_and_keeps_recording_sessions_together(self):
        rows = [{"session": f"day{i // 3}"} for i in range(60)]
        splits = split_by_session(rows)
        self.assertEqual(splits, split_by_session(rows))
        sessions = [{row["session"] for row in splits[name]} for name in splits]
        self.assertTrue(all(sessions))
        for index, left in enumerate(sessions):
            for right in sessions[index + 1:]:
                self.assertFalse(left & right)
        self.assertEqual(sum(map(len, splits.values())), len(rows))
        with self.assertRaises(ValueError):
            split_by_session([{"session": "one"}, {"session": "two"}])

    def test_preparation_detects_same_audio_under_different_filenames(self):
        self.wav("one.wav")
        copy = self.wav("copy.wav")
        content = bytearray(copy.read_bytes())
        content.extend(b"JUNK" + (4).to_bytes(4, "little") + b"test")
        content[4:8] = (len(content) - 8).to_bytes(4, "little")
        copy.write_bytes(content)
        self.assertNotEqual((self.root / "one.wav").read_bytes(), copy.read_bytes())
        source = self.manifest([self.row("one.wav"), self.row("copy.wav", "day2")])
        output = self.root / "prepared"
        with self.assertRaisesRegex(ValueError, "중복 음성"):
            prepare_dataset(source, output)
        self.assertFalse(output.exists())

    def test_preparation_writes_three_splits_and_never_overwrites(self):
        rows = []
        for index in range(3):
            name = f"{index}.wav"
            self.wav(name, frequency=220 + index * 100)
            rows.append(self.row(name, f"day{index}"))
        source = self.manifest(rows)
        output = self.root / "prepared"
        splits = prepare_dataset(source, output)
        for name in ("train", "validation", "test"):
            records = read_manifest(output / f"{name}.jsonl")
            self.assertEqual(records, splits[name])
            self.assertEqual(len(records[0]["audio_sha256"]), 64)
        with self.assertRaises(FileExistsError):
            prepare_dataset(source, output)

    def test_stereo_resampling_produces_finite_16khz_mono(self):
        path = self.wav("stereo.wav", rate=48000, channels=2)
        audio = load_audio(path)
        self.assertEqual(audio.shape, (1600,))
        self.assertEqual(audio.dtype, np.float32)
        self.assertTrue(np.isfinite(audio).all())
        self.assertAlmostEqual(validate_audio(path)[0], .1)

    def test_overlong_and_empty_recordings_are_rejected(self):
        for name, duration in (("long.wav", 30.01), ("empty.wav", 0)):
            path = self.wav(name, seconds=duration)
            with self.subTest(name=name), self.assertRaises(ValueError):
                load_audio(path)

    def test_nonfinite_audio_is_rejected(self):
        import soundfile as sf
        path = self.root / "invalid.wav"
        sf.write(path, np.array([0., np.nan, np.inf], dtype="float32"),
                 16000, subtype="FLOAT")
        with self.assertRaisesRegex(ValueError, "NaN"):
            load_audio(path)

    def prepared_v1(self):
        rows = []
        for index in range(3):
            name = f"old-{index}.wav"
            self.wav(name, frequency=300 + index * 100)
            rows.append(self.row(name, f"day{index}"))
        prior = self.root / "v1"
        splits = prepare_dataset(self.manifest(rows), prior)
        return rows, prior, splits

    def test_continued_preparation_preserves_splits_and_adds_new_session_to_train(self):
        rows, prior, old = self.prepared_v1()
        self.wav("new.wav", frequency=900)
        rows.append(self.row("new.wav", "new-day"))
        new = prepare_dataset(self.manifest(rows), self.root / "v2", seed=999,
                              previous_data=prior)
        for name in ("validation", "test"):
            self.assertEqual(new[name], old[name])
        self.assertEqual({row["session"] for row in new["train"]},
                         {old["train"][0]["session"], "new-day"})

    def test_continued_preparation_rejects_inconsistent_prior_session_assignments(self):
        rows, prior, old = self.prepared_v1()
        (prior / "validation.jsonl").write_text(
            json.dumps(old["train"][0], ensure_ascii=False) + "\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "여러 분할"):
            prepare_dataset(self.manifest(rows), self.root / "v2", previous_data=prior)
        self.assertFalse((self.root / "v2").exists())

    def test_continued_preparation_requires_all_previous_heldout_recordings(self):
        rows, prior, old = self.prepared_v1()
        missing = old["test"][0]["session"]
        rows = [row for row in rows if row["session"] != missing]
        with self.assertRaisesRegex(ValueError, "모든 음성과 정답"):
            prepare_dataset(self.manifest(rows), self.root / "v2", previous_data=prior)


if __name__ == "__main__":
    unittest.main()
