"""Exercise waveform selection and exported clips using the real Tk editor."""
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import Mock, patch
import wave

import numpy as np
import soundfile as sf

from clip_audio import export_segments, prepare_audio

try:
    import tkinter as tk
except ImportError:
    tk = None
if tk is not None:
    from audio_editor import AudioEditor


@unittest.skipIf(tk is None, "Tkinter is not installed")
class AudioEditorTests(unittest.TestCase):
    def setUp(self):
        try:
            self.root = tk.Tk()
        except tk.TclError as error:
            self.skipTest(f"GUI display required: {error}")
        self.root.geometry("240x120")
        self.temp = tempfile.TemporaryDirectory()
        self.folder = Path(self.temp.name)
        self.source = self.folder / "긴 원본 녹음.wav"
        sf.write(self.source, np.sin(np.arange(45 * 16000) * .037) * .15, 16000)
        self.original = self.source.read_bytes()
        self.audio = prepare_audio(self.source)
        self.order, self.saved, self.closed_with_temp = [], [], []
        self.play = Mock()
        self.stop = Mock(side_effect=lambda: self.order.append("stop"))

        def on_close():
            self.order.append("close")
            self.closed_with_temp.append(self.audio.path.exists())

        def on_saved(paths, session, language, add_to_list):
            self.order.append("saved")
            self.saved.append((paths, session, language, add_to_list))

        self.editor = AudioEditor(self.root, self.audio, language="en", on_saved=on_saved,
                                  on_close=on_close, play=self.play, stop=self.stop)
        self.error = patch("audio_editor.messagebox.showerror").start()
        self.warning = patch("audio_editor.messagebox.showwarning").start()
        self.root.update()

    def tearDown(self):
        if hasattr(self, "editor"):
            self.editor.close()
            self.audio.close()
            self.root.destroy()
            self.temp.cleanup()
            patch.stopall()

    def wait_for(self, predicate):
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            self.root.update()
            if predicate():
                return
            time.sleep(.01)
        self.fail("Audio editor operation did not finish")

    def select(self, start, end):
        self.editor.start_var.set(str(start))
        self.editor.end_var.set(str(end))

    def test_drag_manual_edit_limit_and_cancel_preserve_the_source(self):
        self.editor.language_input.focus_force()
        for language, sequence, options in (("en", "<MouseWheel>", {"delta": -120}),
                                             ("ko", "<MouseWheel>", {"delta": 120}),
                                             ("en", "<Button-5>", {}),
                                             ("ko", "<Button-4>", {})):
            self.editor.language.set(language)
            self.editor.language_input.event_generate(sequence, **options)
            self.root.update()
            self.assertEqual(self.editor.language.get(), language)
        width = self.editor.canvas.winfo_width()
        self.assertGreater(width, 100)
        self.editor.canvas.event_generate("<ButtonPress-1>", x=width // 4, y=40)
        self.editor.canvas.event_generate("<B1-Motion>", x=width // 2, y=40)
        self.editor.canvas.event_generate("<ButtonRelease-1>", x=width // 2, y=40)
        self.root.update()
        self.assertGreater(float(self.editor.start_var.get()), 0)
        self.assertGreater(float(self.editor.end_var.get()), float(self.editor.start_var.get()))
        self.assertLess(float(self.editor.end_var.get()), self.audio.duration)

        self.select(2.25, 6.5)
        self.editor.add_segment()
        self.assertEqual(self.editor.segments, [(2.25, 6.5)])
        self.editor.segment_list.selection_set("0")
        self.root.update()
        self.select(3, 5)
        self.editor.update_segment()
        self.assertEqual(self.editor.segments, [(3, 5)])
        self.select(0, 31)
        self.editor.add_segment()
        self.assertEqual(self.editor.segments, [(3, 5)])
        self.assertTrue(self.error.called or self.warning.called)

        self.select(3, 5)
        self.editor.preview_selection()
        self.wait_for(lambda: self.play.called)
        self.play.assert_called_once()
        with wave.open(str(self.play.call_args.args[0]), "rb") as recording:
            self.assertEqual(recording.getnframes(), 2 * 16000)
        self.editor.close()
        self.editor.close()
        self.assertEqual(self.closed_with_temp, [False])
        self.assertEqual(self.saved, [])
        self.assertEqual(self.order[-1], "close")
        self.assertIn("stop", self.order[:-1])
        self.assertFalse(self.audio.path.exists())
        self.assertEqual(self.source.read_bytes(), self.original)

    def test_multiple_clips_save_before_callback_with_one_source_session(self):
        self.editor.geometry("820x640")
        self.root.update()
        right = self.editor.winfo_rootx() + self.editor.winfo_width()
        bottom = self.editor.winfo_rooty() + self.editor.winfo_height()
        for widget in self.editor.buttons + [self.editor.add_check, self.editor.status_label]:
            self.assertTrue(widget.winfo_ismapped())
            self.assertLessEqual(widget.winfo_rootx() + widget.winfo_width(), right)
            self.assertLessEqual(widget.winfo_rooty() + widget.winfo_height(), bottom)
        self.assertTrue(self.editor.add_to_list.get())
        self.assertEqual(self.editor.session.get(), self.audio.session)
        self.select(1, 3)
        self.editor.add_segment()
        self.select(10, 12.5)
        self.editor.add_segment()
        output = self.folder / "저장할 구간"
        self.editor.output_dir.set(str(output))
        self.editor.save_segments()
        self.wait_for(lambda: bool(self.saved))
        self.assertEqual(len(self.saved), 1)
        paths, session, language, add_to_list = self.saved[0]
        self.assertEqual(len(paths), 2)
        self.assertEqual(session, self.audio.session)
        self.assertEqual(language, "en")
        self.assertIs(add_to_list, True)
        for path, frames in zip(paths, (2 * 16000, int(2.5 * 16000))):
            self.assertEqual(Path(path).parent, output)
            with wave.open(str(path), "rb") as recording:
                self.assertEqual(recording.getnframes(), frames)
                self.assertEqual((recording.getnchannels(), recording.getsampwidth(), recording.getframerate()),
                                 (1, 2, 16000))
        self.assertEqual(self.closed_with_temp, [False])
        self.assertEqual(self.order[-2:], ["close", "saved"])
        self.assertEqual(self.source.read_bytes(), self.original)
        self.error.assert_not_called()

    def test_failed_save_keeps_the_editor_open_and_can_retry_without_import(self):
        self.select(0, 2)
        self.editor.add_segment()
        blocked = self.folder / "파일을 폴더로 사용할 수 없음"
        blocked.write_bytes(b"keep existing file")
        self.editor.output_dir.set(str(blocked))
        self.editor.save_segments()
        self.wait_for(lambda: self.error.called)
        self.assertTrue(self.editor.winfo_exists())
        self.assertTrue(self.audio.path.is_file())
        self.assertEqual(self.saved, [])
        self.assertEqual(self.closed_with_temp, [])
        self.assertEqual(blocked.read_bytes(), b"keep existing file")
        self.assertEqual(self.source.read_bytes(), self.original)

        self.editor.output_dir.set(str(self.folder / "다시 저장"))
        self.editor.add_to_list.set(False)
        self.editor.save_segments()
        self.wait_for(lambda: bool(self.saved))
        self.assertEqual(len(self.saved[0][0]), 1)
        self.assertIs(self.saved[0][3], False)
        self.assertEqual(self.closed_with_temp, [False])
        self.assertEqual(self.order[-2:], ["close", "saved"])

    def test_close_during_export_waits_before_cleaning_temporary_audio(self):
        self.select(1, 3)
        self.editor.add_segment()
        self.editor.output_dir.set(str(self.folder / "작업 중 닫기"))
        started, release = threading.Event(), threading.Event()

        def delayed_export(*args):
            started.set()
            if not release.wait(3):
                raise TimeoutError("Test export was not released")
            return export_segments(*args)

        with patch("audio_editor.export_segments", side_effect=delayed_export):
            try:
                self.editor.save_segments()
                self.wait_for(started.is_set)
                self.editor.close()
                self.assertTrue(self.editor.winfo_exists())
                self.assertTrue(self.audio.path.is_file())
                self.assertEqual(self.closed_with_temp, [])
                self.assertEqual(self.saved, [])
            finally:
                release.set()
            self.wait_for(lambda: bool(self.saved))
        self.assertEqual(self.closed_with_temp, [False])
        self.assertEqual(len(self.saved), 1)
        self.assertEqual(self.order[-2:], ["close", "saved"])
        self.assertEqual(self.source.read_bytes(), self.original)
        self.error.assert_not_called()


if __name__ == "__main__":
    unittest.main()
