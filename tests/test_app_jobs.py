import json
import math
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest
import wave
from unittest.mock import Mock, patch

import app_jobs

class AppJobsTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.project = Path(temporary.name) / "한글 project"
        self.prepared = self.project / "prepared split"
        self.prepared.mkdir(parents=True)
        for split in ("train", "validation", "test"):
            (self.prepared / f"{split}.jsonl").write_text("{}\n", encoding="utf-8")

    def test_prepare_passes_previous_split_and_uses_unique_destination(self):
        (self.project / "manifest.jsonl").write_text("{}\n", encoding="utf-8")
        external = self.project / "별도 검증.jsonl"
        external.write_text("{}\n", encoding="utf-8")
        with patch("prepare.prepare_dataset") as prepare:
            first = app_jobs.prepare_project(self.project, self.prepared,
                                             validation_manifest=external)
            second = app_jobs.prepare_project(self.project)
        self.assertNotEqual(first, second)
        self.assertEqual(first.parent, self.project / "prepared")
        self.assertEqual(prepare.call_args_list[0].args, (self.project / "manifest.jsonl", first))
        self.assertEqual(prepare.call_args_list[0].kwargs["previous_data"], self.prepared)
        self.assertEqual(prepare.call_args_list[0].kwargs["validation_manifest"], external)

    def test_external_validation_checks_adapter_history_and_preserves_prior_files(self):
        from data import read_manifest, validate_audio
        from prepare import prepare_dataset

        def recording(name, frequency):
            path = self.project / name
            values = [int(4000 * math.sin(index * 2 * math.pi * frequency / 16000))
                      for index in range(1600)]
            with wave.open(str(path), "wb") as stream:
                stream.setnchannels(1)
                stream.setsampwidth(2)
                stream.setframerate(16000)
                stream.writeframes(struct.pack(f"<{len(values)}h", *values))
            return path

        def row(audio, session):
            return {"audio": str(audio), "session": session, "text": "검수한 정답",
                    "language": "ko", "reviewed": True}

        main = self.project / "manifest.jsonl"
        rows = [row(recording(f"old-{index}.wav", 300 + index * 100), f"day-{index}")
                for index in range(3)]
        main.write_text("".join(json.dumps(item, ensure_ascii=False) + "\n" for item in rows),
                        encoding="utf-8")
        prior = self.project / "prior"
        prepare_dataset(main, prior)
        historical_audio = recording("historical.wav", 1700)
        fresh_audio = recording("external.wav", 1900)
        history = {
            "training_sessions": ["historical-training"],
            "selection_sessions": ["historical-training"],
            "held_out_sessions": [],
            "training_audio_sha256": [validate_audio(historical_audio)[1]],
            "selection_audio_sha256": [validate_audio(historical_audio)[1]],
            "held_out_audio_sha256": [],
        }
        adapter = self.project / "model"
        adapter.mkdir()
        (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
        (adapter / "training_info.json").write_text(json.dumps(history), encoding="utf-8")
        original = {path: path.read_bytes() for path in
                    [main, *prior.iterdir(), *adapter.iterdir(), *self.project.glob("*.wav")]}
        external = self.project / "external.jsonl"
        for name, candidate in (
                ("trained-session", row(fresh_audio, "historical-training")),
                ("trained-audio", row(historical_audio, "new-validation-session"))):
            with self.subTest(case=name):
                external.write_text(json.dumps(candidate, ensure_ascii=False) + "\n", encoding="utf-8")
                before = external.read_bytes()
                with patch("runtime.adapter_source", return_value=("openai/whisper-small", None, history)):
                    with self.assertRaisesRegex(ValueError, "이전 모델의 학습 자료"):
                        app_jobs.prepare_project(self.project, prior,
                                                 validation_manifest=external, adapter=adapter)
                self.assertEqual(external.read_bytes(), before)
                self.assertFalse(list((self.project / "prepared").glob("*")))
                for path, content in original.items():
                    self.assertEqual(path.read_bytes(), content)

        external.write_text(json.dumps(row(fresh_audio, "new-validation-session"), ensure_ascii=False) + "\n",
                            encoding="utf-8")
        with patch("runtime.adapter_source", return_value=("openai/whisper-small", None, history)):
            prepared = app_jobs.prepare_project(self.project, prior,
                                                validation_manifest=external, adapter=adapter)
        validation = read_manifest(prepared / "validation.jsonl")
        self.assertIn("new-validation-session", {item["session"] for item in validation})
        for path, content in original.items():
            self.assertEqual(path.read_bytes(), content)

    def test_training_uses_safe_unicode_arguments_and_unique_run_paths(self):
        adapter = self.project / "이전 model"
        adapter.mkdir()
        for filename in ("adapter_config.json", "training_info.json"):
            (adapter / filename).write_text("{}", encoding="utf-8")
        relative = Path(os.path.relpath(self.project))
        with patch("app_jobs.subprocess.Popen") as popen:
            _, first = app_jobs.start_training(relative, relative / "prepared split", "cpu", adapter, quick=True)
            _, second = app_jobs.start_training(self.project, self.prepared)
        command = popen.call_args_list[0].args[0]
        options = popen.call_args_list[0].kwargs
        self.assertNotEqual(first, second)
        self.assertTrue(first.is_absolute())
        self.assertEqual(command[:3], [sys.executable, "-u", str(Path(app_jobs.__file__).with_name("train.py"))])
        self.assertEqual(command[command.index("--data") + 1], str(self.prepared))
        self.assertEqual(command[command.index("--adapter") + 1], str(adapter))
        self.assertEqual(command[-4:], ["--epochs", "1", "--max-steps", "1"])
        self.assertFalse(options.get("shell", False))
        self.assertEqual(options["cwd"], Path(app_jobs.__file__).parent)
        self.assertEqual((options["stdout"], options["stderr"]), (subprocess.PIPE, subprocess.STDOUT))
        self.assertEqual((options["text"], options["encoding"], options["errors"]), (True, "utf-8", "replace"))
        self.assertEqual(options["env"]["PYTHONIOENCODING"], "utf-8")

        (self.prepared / "test.jsonl").unlink()
        with patch("app_jobs.subprocess.Popen") as popen:
            with self.assertRaises((ValueError, FileNotFoundError)):
                app_jobs.start_training(self.project, self.prepared)
            popen.assert_not_called()

    def test_dependencies_and_stopping(self):
        with patch("importlib.util.find_spec", side_effect=lambda name: None if name == "peft" else object()):
            self.assertEqual(app_jobs.training_dependencies(), ["peft"])
        for finished in (False, True):
            process = Mock()
            process.poll.return_value = 0 if finished else None
            process.wait.side_effect = [subprocess.TimeoutExpired("train", 3), 0]
            app_jobs.stop_training(process)
            self.assertEqual(process.terminate.call_count, int(not finished))
            self.assertEqual(process.kill.call_count, int(not finished))
            process.stdout.close.assert_called_once()

    def test_failed_stop_does_not_close_a_running_process_log(self):
        process = Mock()
        process.poll.return_value = None
        process.terminate.side_effect = PermissionError("still running")
        with self.assertRaises(PermissionError):
            app_jobs.stop_training(process)
        process.stdout.close.assert_not_called()
