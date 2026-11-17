"""Exercise the real desktop widgets and save/train boundary on a display."""
import io
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import Mock, patch

import numpy as np
import soundfile as sf

from review_store import load_project

try:
    import tkinter as tk
except ImportError:
    tk = None
if tk is not None:
    from app import ReviewApp

@unittest.skipIf(tk is None, "Tkinter is not installed")
class DesktopTests(unittest.TestCase):
    def setUp(self):
        try:
            self.root = tk.Tk()
        except tk.TclError as error:
            self.skipTest(f"GUI display required: {error}")
        self.root.withdraw()
        self.temporary = tempfile.TemporaryDirectory()
        self.folder = Path(self.temporary.name)
        self.project = self.folder / "my voice"
        self.app = ReviewApp(self.root, self.project)
        self.warning = patch("app.messagebox.showwarning").start()
        self.error = patch("app.messagebox.showerror").start()
        self.wait()

    def tearDown(self):
        if hasattr(self, "app"):
            self.app.destroy()
            self.temporary.cleanup()
            patch.stopall()

    def wait(self):
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            self.root.update()
            if not self.app.busy and self.app.events.empty():
                return
            time.sleep(.01)
        self.fail("Desktop operation did not finish")

    def add_three(self):
        for index in range(3):
            source = self.folder / f"녹음 {index}.wav"
            sf.write(source, np.sin(np.arange(1600) * (.03 + .02 * index)) * .1, 16000)
            self.app.session.set(f"session-{index}")
            with patch("app.filedialog.askopenfilenames", return_value=(str(source),)):
                self.app.add_files()
            self.wait()
        self.assertEqual(len(self.app.views), 3)

    def test_individual_answers_bulk_save_and_draft_restore(self):
        self.add_three()
        for view, answer in zip(self.app.views, ("지금 시작합니다.", "마이크가 잘 들리나요?")):
            view.text.insert("1.0", answer)
        self.root.update()
        self.app.mark_all_reviewed()
        self.app.save()
        self.wait()
        restored = load_project(self.project)
        self.assertEqual([row["text"] for row in restored], ["지금 시작합니다.", "마이크가 잘 들리나요?", ""])
        manifest = [json.loads(line) for line in (self.project / "manifest.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertEqual(len(manifest), 2)
        self.assertEqual(len({row["session"] for row in restored}), 3)
        self.assertFalse(restored[-1]["reviewed"])
        self.error.assert_not_called()

    def test_common_answer_only_fills_blank_and_edit_invalidates_review(self):
        self.add_three()
        self.app.views[0].text.insert("1.0", "개별 정답")
        self.app.common_answer.set("공통 정답")
        self.app.apply_common()
        self.app.mark_all_reviewed()
        self.root.update()
        self.assertEqual([view.record["text"] for view in self.app.views], ["개별 정답", "공통 정답", "공통 정답"])
        self.assertTrue(all(view.reviewed.get() for view in self.app.views))
        self.app.views[1].text.insert("end", " 수정")
        self.root.update()
        self.assertFalse(self.app.views[1].reviewed.get())
        self.assertTrue(self.app.views[0].reviewed.get())

    def test_training_prepares_current_answers_and_streams_process_log(self):
        self.add_three()
        self.app.common_answer.set("새로 입력한 정답")
        self.app.apply_common()
        self.app.mark_all_reviewed()
        output = self.project / "runs" / "test-run"
        (output / "best").mkdir(parents=True)

        class FinishedProcess:
            stdout = io.StringIO("training-step=1\n")

            def wait(self):
                return 0

        def start(project, prepared, device, adapter, quick):
            rows = [json.loads(line) for name in ("train", "validation", "test")
                    for line in (prepared / f"{name}.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual({row["text"] for row in rows}, {"새로 입력한 정답"})
            self.assertTrue(quick)
            return FinishedProcess(), output

        with patch("app.training_dependencies", return_value=[]), patch("app.start_training", side_effect=start) as launcher:
            self.app.train(quick=True)
            self.wait()
        launcher.assert_called_once()
        self.assertIn("training-step=1", self.app.log.get("1.0", "end"))
        self.assertEqual(self.app.adapter.get(), str(output / "best"))
        self.assertTrue(Path(self.app.previous_data.get()).is_dir())
        self.error.assert_not_called()

    def test_failed_stop_restores_controls_and_keeps_the_window_open(self):
        self.app.process = Mock()
        self.app.process.poll.return_value = None
        self.app.stop_requested = self.app.close_after_stop = True
        self.app.events.put(("stop_error", "still running"))
        self.wait()
        self.assertFalse(self.app.stop_requested)
        self.assertFalse(self.app.close_after_stop)
        self.assertEqual(str(self.app.stop_button.cget("state")), "normal")

    def test_status_and_save_controls_remain_visible_at_minimum_window_size(self):
        self.add_three()
        self.root.deiconify()
        self.root.geometry("980x720")
        self.root.update()
        bottom = self.app.status_label.winfo_rooty() + self.app.status_label.winfo_height()
        self.assertLessEqual(bottom, self.root.winfo_rooty() + self.root.winfo_height())
        for button in self.app.actions:
            self.assertLessEqual(button.winfo_rooty() + button.winfo_height(), self.root.winfo_rooty() + self.root.winfo_height())


if __name__ == "__main__":
    unittest.main()
