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
from app_jobs import prepare_project

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

    def test_new_file_language_and_explicit_visible_language_changes_are_preserved(self):
        self.app.language.set("ko")
        self.add_three()
        self.assertTrue(all(view.language_label.winfo_class() == "TLabel" for view in self.app.views))
        self.app.common_answer.set("검수한 정답")
        self.app.apply_common()
        self.app.mark_all_reviewed()
        self.app.save()
        self.wait()
        saved = load_project(self.project)
        self.app.hide_record(saved[0]["id"])

        self.app.language.set("en")
        self.assertEqual(self.app.records, saved)
        self.assertFalse(self.app.dirty)
        self.app.apply_language()
        self.assertEqual([row["language"] for row in self.app.records], ["ko", "en", "en"])
        self.assertEqual([row["reviewed"] for row in self.app.records], [True, False, False])
        self.assertTrue(self.app.dirty)
        self.app.mark_all_reviewed()
        self.app.save()
        self.wait()
        self.app.apply_language()
        self.assertFalse(self.app.dirty)
        self.assertTrue(all(row["reviewed"] for row in self.app.records))

        self.app.language.set("ko")
        self.assertEqual([row["language"] for row in self.app.records], ["ko", "en", "en"])
        source = self.folder / "새 한국어.wav"
        sf.write(source, np.sin(np.arange(1600) * .13) * .1, 16000)
        self.app.session.set("new-ko-session")
        with patch("app.filedialog.askopenfilenames", return_value=(str(source),)):
            self.app.add_files()
        self.wait()
        self.assertEqual([row["language"] for row in self.app.records], ["ko", "en", "en", "ko"])
        self.app.views[-1].text.insert("1.0", "새 한국어 정답")
        self.app.mark_all_reviewed()
        self.app.save()
        self.wait()
        self.app.open_project(self.project, ask=False)
        self.wait()
        restored = load_project(self.project)
        self.assertEqual(self.app.records, restored)
        self.assertEqual([row["language"] for row in restored], ["ko", "en", "en", "ko"])
        self.assertTrue(all(row["reviewed"] for row in restored))
        self.error.assert_not_called()

    def test_mouse_wheel_does_not_change_the_new_file_language(self):
        self.root.deiconify()
        self.root.update()
        self.app.language_input.focus_force()
        for language, sequence, options in (
                ("ko", "<MouseWheel>", {"delta": -120}),
                ("en", "<MouseWheel>", {"delta": 120}),
                ("ko", "<Button-5>", {}),
                ("en", "<Button-4>", {})):
            with self.subTest(language=language, sequence=sequence):
                self.app.language.set(language)
                self.app.language_input.event_generate(sequence, **options)
                self.root.update()
                self.assertEqual(self.app.language.get(), language)

    def test_hidden_saved_session_survives_new_session_save_and_reopen(self):
        self.add_three()
        for index, view in enumerate(self.app.views):
            view.session.set("phone-01")
            view.text.insert("1.0", f"휴대폰 정답 {index}")
        self.app.mark_all_reviewed()
        self.app.save()
        self.wait()
        saved = load_project(self.project)
        audio_before = {path: path.read_bytes() for path in (self.project / "audio").glob("*.wav")}

        self.app.hide_record(saved[0]["id"])
        self.assertEqual(len(self.app.views), 2)
        self.assertFalse(self.app.dirty)
        self.app.show_all()
        self.assertEqual(len(self.app.views), 3)
        self.assertFalse(self.app.dirty)
        self.app.hide_visible()
        self.assertEqual(len(self.app.views), 0)
        self.assertFalse(self.app.dirty)

        source = self.folder / "새 마이크.wav"
        sf.write(source, np.sin(np.arange(1600) * .11) * .1, 16000)
        original = source.read_bytes()
        self.app.session.set("mic-01")
        with patch("app.filedialog.askopenfilenames", return_value=(str(source),)):
            self.app.add_files()
        self.wait()
        self.assertEqual(len(self.app.views), 1)
        self.app.views[0].text.insert("1.0", "새 마이크 정답")
        self.app.mark_all_reviewed()
        self.app.save()
        self.wait()

        restored = load_project(self.project)
        self.assertEqual(restored[:3], saved)
        self.assertEqual(restored[3]["text"], "새 마이크 정답")
        self.assertEqual(restored[3]["session"], "mic-01")
        self.assertTrue(all(row["reviewed"] for row in restored))
        self.assertEqual(source.read_bytes(), original)
        for path, content in audio_before.items():
            self.assertEqual(path.read_bytes(), content)
        self.assertEqual(len((self.project / "manifest.jsonl").read_text(encoding="utf-8").splitlines()), 4)

        self.app.hide_visible()
        self.assertFalse(self.app.dirty)
        self.app.open_project(self.project, ask=False)
        self.wait()
        self.assertEqual(len(self.app.views), 4)
        self.assertEqual(self.app.records, restored)
        self.assertFalse(self.app.dirty)
        self.error.assert_not_called()

    def test_excluding_data_requires_confirmation_and_keeps_audio(self):
        self.add_three()
        self.app.common_answer.set("검수한 정답")
        self.app.apply_common()
        self.app.mark_all_reviewed()
        self.app.save()
        self.wait()
        saved = load_project(self.project)
        identity = saved[0]["id"]
        audio = self.project / saved[0]["audio"]
        original = audio.read_bytes()

        with patch("app.messagebox.askyesno", return_value=False) as confirmation:
            self.app.remove(identity)
        confirmation.assert_called_once()
        self.assertEqual(self.app.records, saved)
        self.assertFalse(self.app.dirty)
        self.assertEqual(load_project(self.project), saved)

        with patch("app.messagebox.askyesno", return_value=True) as confirmation:
            self.app.remove(identity)
        confirmation.assert_called_once()
        self.assertEqual(self.app.records, saved[1:])
        self.assertTrue(self.app.dirty)
        self.assertEqual(load_project(self.project), saved)
        self.app.save()
        self.wait()
        self.assertEqual(load_project(self.project), saved[1:])
        self.assertEqual(audio.read_bytes(), original)

    def test_training_prepares_current_answers_and_streams_process_log(self):
        self.add_three()
        self.app.common_answer.set("새로 입력한 정답")
        self.app.apply_common()
        self.app.mark_all_reviewed()
        self.app.save()
        self.wait()
        saved = load_project(self.project)
        prior = prepare_project(self.project)
        old_test = prior / "test.jsonl"
        old_test_rows = [json.loads(line) for line in old_test.read_text(encoding="utf-8").splitlines()]
        old_test_rows[0]["language"] = "en"
        old_test.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in old_test_rows),
                            encoding="utf-8")
        prior_before = old_test.read_bytes()
        self.app.previous_data.set(str(prior))
        self.app.hide_visible()
        with patch("app.filedialog.askopenfilenames", return_value=(str(self.folder / "녹음 0.wav"),)):
            self.app.add_files()
        self.wait()
        self.assertEqual(len(self.app.views), 0)
        self.assertEqual(self.app.records, saved)
        self.warning.assert_called_once()
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
            self.assertEqual({row["session"] for row in rows}, {"session-0", "session-1", "session-2"})
            corrections = json.loads((prepared / "label_corrections.json").read_text(encoding="utf-8"))
            self.assertEqual(len(corrections["corrections"]), 1)
            self.assertEqual(corrections["corrections"][0]["before"]["language"], "en")
            self.assertEqual(corrections["corrections"][0]["after"]["language"], "ko")
            self.assertEqual(old_test.read_bytes(), prior_before)
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
