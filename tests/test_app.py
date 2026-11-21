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

    def wait_until(self, condition):
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            self.root.update()
            if condition():
                return
            time.sleep(.01)
        self.fail("Desktop condition did not become ready")

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

    def test_play_path_and_existing_record_playback_show_the_correct_name(self):
        path = self.folder / "편집용 임시.wav"
        with patch("app.sys.platform", "linux"), \
                patch("app.shutil.which", return_value="/mock/ffplay"), \
                patch("app.subprocess.Popen") as player:
            self.app.play_path(path)
            self.assertEqual(self.app.status.get(), f"재생 중: {path.name}")
            self.assertEqual(player.call_args.args[0][-1], str(path))
            self.app.play({"audio": "audio/stored.wav", "source_name": "원래 파일.m4a"})
            self.assertEqual(self.app.status.get(), "재생 중: 원래 파일.m4a")
            self.assertEqual(player.call_args.args[0][-1], str(self.project / "audio/stored.wav"))
        self.error.assert_not_called()

    def test_audio_editor_import_preserves_existing_hidden_and_reviewed_data(self):
        self.add_three()
        self.app.common_answer.set("보존할 기존 정답")
        self.app.apply_common()
        self.app.mark_all_reviewed()
        self.app.views[1].language.set("en")
        self.app.mark_all_reviewed()
        self.app.save()
        self.wait()
        saved = load_project(self.project)
        hidden = saved[0]["id"]
        self.app.hide_record(hidden)
        original_audio = {path: path.read_bytes() for path in (self.project / "audio").glob("*.wav")}
        clip = self.folder / "편집한 구간.wav"
        sf.write(clip, np.sin(np.arange(2400) * .17) * .1, 16000)
        clip_before = clip.read_bytes()
        prepared, editor = Mock(), Mock()
        opened = {}

        def create_editor(parent, audio, **options):
            self.assertIs(parent, self.root)
            self.assertIs(audio, prepared)
            opened.update(options)
            return editor

        self.app.language.set("ko")
        with patch("app.filedialog.askopenfilename", return_value=str(clip)), \
                patch("clip_audio.prepare_audio", return_value=prepared), \
                patch("audio_editor.AudioEditor", side_effect=create_editor):
            self.app.open_audio_editor()
            self.wait_until(lambda: bool(opened))
        self.assertTrue(self.app.busy)
        self.assertIs(self.app.audio_editor, editor)
        folder_button = next(button for button in self.app.actions
                             if button.cget("text") == "다른 폴더 열기")
        with patch("app.filedialog.askdirectory") as chooser:
            folder_button.invoke()
        chooser.assert_not_called()
        self.assertEqual(str(self.app.session_input.cget("state")), "disabled")
        self.assertEqual(self.app.records, saved)

        prepared.close()
        opened["on_close"]()
        self.assertFalse(self.app.busy)
        self.assertIsNone(self.app.audio_editor)
        opened["on_saved"]([clip, clip, self.folder / "녹음 0.wav"], "edited-session", "en", True)
        self.wait()
        self.assertEqual(self.app.records[:3], saved)
        self.assertEqual(len(self.app.records), 4)
        self.assertEqual(len(self.app.views), 3)
        self.assertNotIn(hidden, {view.record["id"] for view in self.app.views})
        new_row = self.app.records[-1]
        self.assertEqual((new_row["session"], new_row["language"], new_row["text"], new_row["reviewed"]),
                         ("edited-session", "en", "", False))
        self.warning.assert_called_once()
        self.app.save()
        self.wait()
        self.assertEqual(load_project(self.project), self.app.records)
        self.assertEqual(len((self.project / "manifest.jsonl").read_text(encoding="utf-8").splitlines()), 3)
        self.assertEqual(clip.read_bytes(), clip_before)
        for path, content in original_audio.items():
            self.assertEqual(path.read_bytes(), content)
        self.error.assert_not_called()

    def test_audio_editor_cancel_export_only_and_prepare_failure_preserve_project(self):
        self.add_three()
        self.app.common_answer.set("유지할 정답")
        self.app.apply_common()
        self.app.mark_all_reviewed()
        self.app.save()
        self.wait()
        saved = load_project(self.project)
        manifest_before = (self.project / "manifest.jsonl").read_bytes()
        draft_before = (self.project / ".lvit-review.json").read_bytes()
        clip = self.folder / "외부에 저장한 구간.wav"
        sf.write(clip, np.sin(np.arange(2400) * .19) * .1, 16000)
        prepared, opened = Mock(), {}

        def create_editor(parent, audio, **options):
            opened.update(options)
            return Mock()

        with patch("app.filedialog.askopenfilename", return_value=""), \
                patch("clip_audio.prepare_audio") as prepare_call, \
                patch("audio_editor.AudioEditor") as factory:
            self.app.open_audio_editor()
        prepare_call.assert_not_called()
        factory.assert_not_called()
        self.assertFalse(self.app.busy)

        with patch("app.filedialog.askopenfilename", return_value=str(clip)), \
                patch("clip_audio.prepare_audio", return_value=prepared), \
                patch("audio_editor.AudioEditor", side_effect=create_editor):
            self.app.open_audio_editor()
            self.wait_until(lambda: bool(opened))
        prepared.close()
        opened["on_close"]()
        opened["on_saved"]([clip], "not-for-import", "en", False)
        self.assertFalse(self.app.busy)
        self.assertFalse(self.app.dirty)
        self.assertEqual(self.app.records, saved)
        self.assertEqual((self.project / "manifest.jsonl").read_bytes(), manifest_before)
        self.assertEqual((self.project / ".lvit-review.json").read_bytes(), draft_before)

        with patch("app.filedialog.askopenfilename", return_value=str(clip)), \
                patch("clip_audio.prepare_audio", side_effect=ValueError("음성 준비 실패")):
            self.app.open_audio_editor()
            self.wait()
        self.assertFalse(self.app.busy)
        self.assertIsNone(self.app.audio_editor)
        self.assertEqual(self.app.records, saved)
        self.assertEqual(load_project(self.project), saved)
        self.assertIn("음성 준비 실패", self.app.status.get())

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

    def test_validation_connection_saves_current_rows_and_persists_through_training(self):
        self.add_three()
        self.app.common_answer.set("기존 학습 정답")
        self.app.apply_common()
        self.app.mark_all_reviewed()
        self.app.save()
        self.wait()
        previous = prepare_project(self.project)
        previous_bytes = {path: path.read_bytes() for path in previous.iterdir() if path.is_file()}
        adapter = self.project / "runs" / "previous" / "best"
        adapter.mkdir(parents=True)
        source = self.folder / "별도 검증 자료"
        source.mkdir()
        manifest = source / "manifest.jsonl"
        manifest.write_text('{"text":"별도 검증 정답"}\n', encoding="utf-8")
        source_before = manifest.read_bytes()
        (self.project / ".lvit-app.json").write_text(
            json.dumps({"previous_data": str(previous), "adapter": str(adapter)}), encoding="utf-8")
        self.app.validation_manifest.set("stale-link-from-another-project")
        self.app.open_project(self.project, ask=False)
        self.wait()
        self.assertEqual(self.app.validation_manifest.get(), "")
        self.assertEqual(self.app.previous_data.get(), str(previous))
        self.assertEqual(self.app.adapter.get(), str(adapter))
        self.app.hide_record(self.app.records[0]["id"])
        self.app.views[0].text.delete("1.0", "end")
        self.app.views[0].text.insert("1.0", "연결 직전 수정한 정답")
        self.app.mark_all_reviewed()
        self.assertTrue(self.app.dirty)
        linked = self.project / "prepared" / "linked"
        linked.mkdir()
        for name in ("train", "validation", "test"):
            (linked / f"{name}.jsonl").write_bytes((previous / f"{name}.jsonl").read_bytes())

        def prepare(project, prior, *, validation_manifest, adapter):
            self.assertEqual(project, self.project)
            self.assertEqual(prior, str(previous))
            self.assertEqual(validation_manifest, str(manifest))
            self.assertEqual(adapter, self.app.adapter.get())
            saved = load_project(project)
            self.assertEqual(len(saved), 3)
            self.assertEqual(saved[1]["text"], "연결 직전 수정한 정답")
            self.assertTrue(all(row["reviewed"] for row in saved))
            return linked

        with patch("app.training_dependencies", return_value=[]), \
                patch("app.filedialog.askdirectory", return_value=str(source)), \
                patch("app.prepare_project", side_effect=prepare) as prepare_call:
            self.app.connect_validation_folder()
            self.wait()
        prepare_call.assert_called_once()
        self.assertEqual(self.app.previous_data.get(), str(linked))
        self.assertEqual(self.app.adapter.get(), str(adapter))
        self.assertEqual(self.app.validation_manifest.get(), str(manifest))
        self.assertEqual(manifest.read_bytes(), source_before)
        for path, content in previous_bytes.items():
            self.assertEqual(path.read_bytes(), content)
        self.app.open_project(self.project, ask=False)
        self.wait()
        self.assertEqual(self.app.validation_manifest.get(), str(manifest))
        self.assertEqual(self.app.previous_data.get(), str(linked))
        self.assertEqual(self.app.adapter.get(), str(adapter))

        output = self.project / "runs" / "after-validation-link"
        (output / "best").mkdir(parents=True)

        class FinishedProcess:
            stdout = io.StringIO("linked-validation-training\n")

            def wait(self):
                return 0

        with patch("app.training_dependencies", return_value=[]), \
                patch("app.prepare_project", return_value=linked) as prepare_call, \
                patch("app.start_training", return_value=(FinishedProcess(), output)) as launcher:
            self.app.train(quick=True)
            self.wait()
        prepare_call.assert_called_once_with(self.project, str(linked),
                                             validation_manifest=str(manifest), adapter=str(adapter))
        launcher.assert_called_once_with(self.project, linked, "auto", str(adapter), True)
        self.app.open_project(self.project, ask=False)
        self.wait()
        self.assertEqual(self.app.validation_manifest.get(), str(manifest))
        self.assertEqual(self.app.adapter.get(), str(output / "best"))
        self.error.assert_not_called()

    def test_validation_connection_cancel_and_failure_keep_existing_state(self):
        with patch("app.messagebox.showinfo") as info, \
                patch("app.filedialog.askdirectory") as chooser, \
                patch("app.prepare_project") as prepare_call:
            self.app.connect_validation_folder()
        info.assert_called_once()
        chooser.assert_not_called()
        prepare_call.assert_not_called()

        self.add_three()
        self.app.common_answer.set("보존할 학습 정답")
        self.app.apply_common()
        self.app.mark_all_reviewed()
        self.app.save()
        self.wait()
        saved = load_project(self.project)
        previous = prepare_project(self.project)
        adapter = self.project / "runs" / "previous" / "best"
        adapter.mkdir(parents=True)
        old_manifest = self.folder / "old-validation.jsonl"
        old_manifest.write_text("old-validation\n", encoding="utf-8")
        source = self.folder / "실패할 검증 자료"
        source.mkdir()
        manifest = source / "manifest.jsonl"
        manifest.write_text("invalid-validation\n", encoding="utf-8")
        self.app.previous_data.set(str(previous))
        self.app.adapter.set(str(adapter))
        self.app.validation_manifest.set(str(old_manifest))
        self.app._write_state()
        state_before = (self.project / ".lvit-app.json").read_bytes()

        with patch("app.training_dependencies", return_value=[]), \
                patch("app.filedialog.askdirectory", return_value=""), \
                patch("app.prepare_project") as prepare_call:
            self.app.connect_validation_folder()
        prepare_call.assert_not_called()

        with patch("app.training_dependencies", return_value=[]), \
                patch("app.filedialog.askdirectory", return_value=str(source)), \
                patch("app.prepare_project", side_effect=ValueError("검증 자료 연결 실패")):
            self.app.connect_validation_folder()
            self.wait()
        self.assertIn("검증 자료 연결 실패", self.app.status.get())
        self.assertEqual(self.app.previous_data.get(), str(previous))
        self.assertEqual(self.app.adapter.get(), str(adapter))
        self.assertEqual(self.app.validation_manifest.get(), str(old_manifest))
        self.assertEqual((self.project / ".lvit-app.json").read_bytes(), state_before)
        self.assertEqual(load_project(self.project), saved)
        self.assertEqual(manifest.read_text(encoding="utf-8"), "invalid-validation\n")
        self.assertEqual(old_manifest.read_text(encoding="utf-8"), "old-validation\n")

        linked = self.project / "prepared" / "unsaved-link"
        linked.mkdir()
        for name in ("train", "validation", "test"):
            (linked / f"{name}.jsonl").write_bytes((previous / f"{name}.jsonl").read_bytes())
        with patch("app.training_dependencies", return_value=[]), \
                patch("app.filedialog.askdirectory", return_value=str(source)), \
                patch("app.prepare_project", return_value=linked), \
                patch.object(self.app, "_write_state", return_value=False) as write_state:
            self.app.connect_validation_folder()
            self.wait()
        write_state.assert_called_once()
        self.assertEqual(self.app.previous_data.get(), str(previous))
        self.assertEqual(self.app.adapter.get(), str(adapter))
        self.assertEqual(self.app.validation_manifest.get(), str(old_manifest))
        self.assertEqual((self.project / ".lvit-app.json").read_bytes(), state_before)
        self.assertEqual(load_project(self.project), saved)
        self.assertNotIn("검증 폴더 연결 완료", self.app.status.get())

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
