import json
from pathlib import Path
import sys
import tempfile
import unittest
import wave

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from review_store import import_recording, load_project, save_project


class ReviewStoreTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.project = self.root / "project"
        self.addCleanup(self.temp.cleanup)

    def wav(self, name="record.wav", seconds=.2, rate=48000, frequency=220):
        path = self.root / name
        samples = (np.sin(np.arange(int(rate * seconds)) * 2 * np.pi * frequency / rate)
                   * 8000).astype("<i2")
        with wave.open(str(path), "wb") as recording:
            recording.setnchannels(1)
            recording.setsampwidth(2)
            recording.setframerate(rate)
            recording.writeframes(samples.tobytes())
        return path

    def test_edit_save_reload_preserves_exact_text_and_portable_paths(self):
        source = self.wav()
        original = source.read_bytes()
        row = import_recording(source, self.project, "day-one")
        row.update(text="지금 마이크 소리 잘 들리시나요? English 42!", reviewed=True)
        manifest, count = save_project(self.project, [row])
        self.assertEqual(count, 1)
        stored = json.loads(manifest.read_text(encoding="utf-8"))
        self.assertEqual(stored["text"], row["text"])
        self.assertFalse(Path(stored["audio"]).is_absolute())
        self.assertEqual(load_project(self.project), [row])
        self.assertEqual(source.read_bytes(), original)

    def test_unreviewed_rows_are_draft_only_and_blank_reviewed_rows_fail(self):
        row = import_recording(self.wav(), self.project, "")
        manifest, count = save_project(self.project, [row])
        before = manifest.read_bytes()
        draft_before = (self.project / ".lvit-review.json").read_bytes()
        self.assertEqual(count, 0)
        self.assertEqual(before, b"")
        self.assertEqual(load_project(self.project)[0]["text"], "")
        row["reviewed"] = True
        with self.assertRaises(ValueError):
            save_project(self.project, [row])
        self.assertEqual(manifest.read_bytes(), before)
        self.assertEqual((self.project / ".lvit-review.json").read_bytes(), draft_before)

    def test_decoded_duplicate_reuses_one_canonical_wav(self):
        first = import_recording(self.wav("one.wav"), self.project, "day-one")
        second = import_recording(self.wav("two.wav"), self.project, "day-two")
        self.assertEqual(first["id"], second["id"])
        self.assertEqual(first["audio"], second["audio"])
        self.assertEqual(len(list((self.project / "audio").glob("*.wav"))), 1)
        with self.assertRaisesRegex(ValueError, "중복"):
            save_project(self.project, [first, second])

    def test_long_and_empty_recordings_are_rejected_without_project_writes(self):
        for duration in (30.01, 0):
            with self.subTest(duration=duration), self.assertRaises(ValueError):
                import_recording(self.wav(seconds=duration), self.project, "one")
        self.assertFalse(self.project.exists())

    def test_malformed_draft_and_outside_project_paths_are_rejected(self):
        row = import_recording(self.wav(), self.project, "one")
        save_project(self.project, [row])
        draft = self.project / ".lvit-review.json"
        draft.write_text("{broken", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "JSON"):
            load_project(self.project)
        draft.write_text(json.dumps([{**row, "audio": str(self.root / "record.wav")}]),
                         encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "프로젝트 밖"):
            load_project(self.project)

    def test_existing_manifest_without_draft_loads_reviewed_rows(self):
        row = import_recording(self.wav(), self.project, "one")
        row.update(text="확인한 정답입니다.", reviewed=True)
        save_project(self.project, [row])
        (self.project / ".lvit-review.json").unlink()
        loaded = load_project(self.project)
        self.assertEqual(loaded[0]["id"], row["id"])
        self.assertEqual(loaded[0]["text"], row["text"])
        self.assertEqual(loaded[0]["source_name"], row["source_name"])

    def test_corrupt_canonical_file_is_never_overwritten(self):
        source = self.wav()
        row = import_recording(source, self.project, "one")
        destination = self.project / row["audio"]
        destination.write_bytes(b"original-other-content")
        with self.assertRaises(ValueError):
            import_recording(source, self.project, "one")
        self.assertEqual(destination.read_bytes(), b"original-other-content")

    def test_phone_m4a_aac_is_decoded_to_pcm16(self):
        import av
        source = self.root / "phone.m4a"
        rate = 48000
        values = (np.sin(np.arange(rate // 4) * 2 * np.pi * 300 / rate) * .2).astype("float32")
        with av.open(str(source), "w") as container:
            stream = container.add_stream("aac", rate=rate)
            stream.layout = "mono"
            frame = av.AudioFrame.from_ndarray(values.reshape(1, -1), format="fltp", layout="mono")
            frame.sample_rate = rate
            for packet in stream.encode(frame):
                container.mux(packet)
            for packet in stream.encode(None):
                container.mux(packet)
        row = import_recording(source, self.project, "phone-day")
        with wave.open(str(self.project / row["audio"]), "rb") as recording:
            self.assertEqual((recording.getframerate(), recording.getnchannels(),
                              recording.getsampwidth()), (16000, 1, 2))
        self.assertGreater(row["duration"], .2)
        self.assertLess(row["duration"], .4)


if __name__ == "__main__":
    unittest.main()
