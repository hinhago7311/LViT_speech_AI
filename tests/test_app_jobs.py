import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
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
        with patch("prepare.prepare_dataset") as prepare:
            first = app_jobs.prepare_project(self.project, self.prepared)
            second = app_jobs.prepare_project(self.project)
        self.assertNotEqual(first, second)
        self.assertEqual(first.parent, self.project / "prepared")
        self.assertEqual(prepare.call_args_list[0].args, (self.project / "manifest.jsonl", first))
        self.assertEqual(prepare.call_args_list[0].kwargs["previous_data"], self.prepared)

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
