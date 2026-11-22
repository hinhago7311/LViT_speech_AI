"""Small waveform editor for exporting clips from one prepared recording."""
import math
from pathlib import Path
import queue
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from clip_audio import clip_filename, default_clip_name, export_segments, validate_segment
from data import MAX_SECONDS, SAMPLE_RATE


class AudioEditor(tk.Toplevel):
    def __init__(self, parent, audio, on_saved=None, on_close=None, play=None, stop=None):
        super().__init__(parent)
        self.audio = audio
        self.on_saved = on_saved or (lambda paths: None)
        self.on_close = on_close or (lambda: None)
        self.play_callback = play or (lambda path: None)
        self.stop_callback = stop or (lambda: None)
        self.segments = []
        self.events = queue.Queue()
        self.busy = self.close_requested = self.closed = False
        self.poll_job = None
        self.drag_start = None
        self.next_number = 1
        self.buttons, self.entries = [], []
        self.start_var = tk.StringVar(self, value="0")
        self.end_var = tk.StringVar(self, value=str(min(audio.duration, MAX_SECONDS)))
        self.selection_info = tk.StringVar(self)
        self.filename = tk.StringVar(self, value=default_clip_name(audio.source, self.next_number))
        self.disposition = tk.StringVar(self, value="저장")
        self.output_dir = tk.StringVar(self, value=str(Path(audio.source).parent / (Path(audio.source).stem + "_cuts")))
        self.status = tk.StringVar(self, value="파형을 드래그해 구간을 선택하고 「구간 담기」를 누르세요.")
        self.title("음성 구간 편집")
        self.geometry("1000x740")
        self.minsize(820, 640)
        self.transient(parent)
        self._build()
        self.start_var.trace_add("write", self._selection_changed)
        self.end_var.trace_add("write", self._selection_changed)
        self._selection_changed()
        self.protocol("WM_DELETE_WINDOW", self.close)
        self.grab_set()
        self.poll_job = self.after(50, self._poll)

    def _button(self, parent, text, command):
        button = ttk.Button(parent, text=text, command=command)
        self.buttons.append(button)
        return button

    def _build(self):
        outer = ttk.Frame(self, padding=14)
        outer.pack(fill="both", expand=True)
        outer.columnconfigure(0, weight=1)
        outer.rowconfigure(1, weight=1, minsize=90)
        outer.rowconfigure(4, weight=1, minsize=80)
        header = ttk.Frame(outer)
        header.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        ttk.Label(header, text=f"{Path(self.audio.source).name} · 전체 {self.audio.duration:.3f}초",
                  wraplength=760, font=("TkDefaultFont", 12, "bold")).pack(anchor="w")
        ttk.Label(header, text="구간 선택 → 파일명·저장/제외 선택 → 구간 담기. 파일만 저장하며 학습 목록에는 등록하지 않습니다.",
                  wraplength=760).pack(anchor="w", pady=(4, 0))
        self.canvas = tk.Canvas(outer, height=150, background="white", highlightthickness=1, highlightbackground="#cbd5e1")
        self.canvas.grid(row=1, column=0, sticky="nsew")
        self.canvas.bind("<Configure>", self._draw_waveform)
        self.canvas.bind("<ButtonPress-1>", self._drag_begin)
        self.canvas.bind("<B1-Motion>", self._drag_move)
        self.canvas.bind("<ButtonRelease-1>", self._drag_end)
        selected = ttk.Frame(outer)
        selected.grid(row=2, column=0, sticky="ew", pady=7)
        for column, (label, variable) in enumerate((("시작(초)", self.start_var), ("끝(초)", self.end_var))):
            ttk.Label(selected, text=label).pack(side="left", padx=(0 if column == 0 else 12, 5))
            entry = ttk.Entry(selected, textvariable=variable, width=16)
            entry.pack(side="left")
            self.entries.append(entry)
        self.selection_label = ttk.Label(selected, textvariable=self.selection_info)
        self.selection_label.pack(side="left", padx=12)
        actions = ttk.Frame(outer)
        actions.grid(row=3, column=0, sticky="ew", pady=(0, 6))
        for text, command in (("전체 듣기", self.play_all), ("선택 구간 듣기", self.preview_selection),
                              ("재생 중지", self.stop_playback), ("구간 담기", self.add_segment),
                              ("선택 구간 수정", self.update_segment), ("선택 구간 삭제", self.remove_segments)):
            self._button(actions, text, command).pack(side="left", padx=(0, 4))
        listing = ttk.Frame(outer)
        listing.grid(row=4, column=0, sticky="nsew")
        self.segment_list = ttk.Treeview(listing, columns=("name", "start", "end", "duration", "action"), show="headings", height=5)
        for name, label in (("name", "저장할 파일명"), ("start", "시작(초)"), ("end", "끝(초)"),
                            ("duration", "길이(초)"), ("action", "처리")):
            self.segment_list.heading(name, text=label)
            self.segment_list.column(name, width=300 if name == "name" else 90, minwidth=60, anchor="center")
        scrollbar = ttk.Scrollbar(listing, orient="vertical", command=self.segment_list.yview)
        self.segment_list.configure(yscrollcommand=scrollbar.set)
        self.segment_list.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        self.segment_list.bind("<<TreeviewSelect>>", self._load_selected)
        output = ttk.Frame(outer)
        output.grid(row=5, column=0, sticky="ew", pady=8)
        ttk.Label(output, text="저장 폴더").pack(side="left", padx=(0, 6))
        ttk.Entry(output, textvariable=self.output_dir, state="readonly").pack(side="left", fill="x", expand=True)
        self._button(output, "저장 폴더 선택", self.choose_output).pack(side="left", padx=(8, 0))
        metadata = ttk.Frame(outer)
        metadata.grid(row=6, column=0, sticky="ew")
        row = ttk.Frame(metadata)
        row.pack(fill="x")
        ttk.Label(row, text="구간 파일명").pack(side="left", padx=(0, 6))
        filename_entry = ttk.Entry(row, textvariable=self.filename)
        filename_entry.pack(side="left", fill="x", expand=True)
        self.entries.append(filename_entry)
        ttk.Label(row, text="처리").pack(side="left", padx=(14, 6))
        self.disposition_input = ttk.Combobox(row, textvariable=self.disposition, values=("저장", "제외"), state="readonly", width=6)
        self.disposition_input.pack(side="left")
        for event in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
            self.disposition_input.bind(event, lambda _event: "break")
        ttk.Label(metadata, text="목록 항목을 고른 뒤 변경하면 「선택 구간 수정」을 누르세요. 제외·삭제는 원본이나 기존 파일을 지우지 않습니다.",
                  wraplength=760).pack(anchor="w", pady=(4, 0))
        footer = ttk.Frame(outer)
        footer.grid(row=7, column=0, sticky="ew", pady=8)
        ttk.Label(footer, text="「저장」으로 표시한 구간만 각각 WAV 파일로 저장합니다.").pack(side="left")
        ttk.Button(footer, text="닫기", command=self.close).pack(side="right")
        self._button(footer, "저장할 구간 파일 저장", self.save_segments).pack(side="right", padx=8)
        self.status_label = ttk.Label(outer, textvariable=self.status, wraplength=760)
        self.status_label.grid(row=8, column=0, sticky="ew")

    def _selection_changed(self, *_args):
        try:
            start, end = float(self.start_var.get()), float(self.end_var.get())
            if not math.isfinite(start) or not math.isfinite(end) or not 0 <= start < end <= self.audio.duration:
                raise ValueError
        except ValueError:
            self.selection_info.set("올바른 시작·끝 초를 입력하세요.")
            self.selection_label.configure(foreground="#b42318")
        else:
            over = round(end * SAMPLE_RATE) - round(start * SAMPLE_RATE) > MAX_SECONDS * SAMPLE_RATE
            self.selection_info.set(f"선택 {end - start:.3f}초" + (" · 30초 초과" if over else ""))
            self.selection_label.configure(foreground="#b42318" if over else "#147a78")
        self._draw_selection()

    def _draw_waveform(self, _event=None):
        self.canvas.delete("waveform")
        width, height = max(1, self.canvas.winfo_width()), max(1, self.canvas.winfo_height())
        peaks = self.audio.peaks
        columns = min(len(peaks), max(1, width - 16))
        for column in range(columns):
            amplitude = max(peaks[column * len(peaks) // columns:(column + 1) * len(peaks) // columns])
            x = 8 + column * max(1, width - 16) / max(1, columns - 1)
            half = max(0, height / 2 - 10) * amplitude
            self.canvas.create_line(x, height / 2 - half, x, height / 2 + half, fill="#315b75", tags="waveform")
        self._draw_selection()

    def _draw_selection(self):
        self.canvas.delete("selection")
        try:
            start, end = float(self.start_var.get()), float(self.end_var.get())
            if not math.isfinite(start) or not math.isfinite(end) or start >= end:
                return
        except ValueError:
            return
        width = max(1, self.canvas.winfo_width() - 16)
        left = 8 + min(1, max(0, start / self.audio.duration)) * width
        right = 8 + min(1, max(0, end / self.audio.duration)) * width
        self.canvas.create_rectangle(left, 1, right, self.canvas.winfo_height() - 1,
                                     fill="#d9eee9", outline="#147a78", width=2, tags="selection")
        self.canvas.tag_lower("selection")

    def _seconds_at(self, x):
        fraction = min(1, max(0, (x - 8) / max(1, self.canvas.winfo_width() - 16)))
        frame = round(fraction * self.audio.frames)
        return self.audio.duration if frame == self.audio.frames else frame / SAMPLE_RATE

    def _drag_begin(self, event):
        if not self.busy:
            self.drag_start = self._seconds_at(event.x)
            self.start_var.set(str(self.drag_start))
            self.end_var.set(str(self.drag_start))

    def _drag_move(self, event):
        if not self.busy and self.drag_start is not None:
            value = self._seconds_at(event.x)
            self.start_var.set(str(min(self.drag_start, value)))
            self.end_var.set(str(max(self.drag_start, value)))

    def _drag_end(self, event):
        self._drag_move(event)
        self.drag_start = None

    def _range(self):
        first, last = validate_segment(self.audio, self.start_var.get(), self.end_var.get())
        return first / SAMPLE_RATE, self.audio.duration if last == self.audio.frames else last / SAMPLE_RATE

    def _error(self, text):
        self.status.set(str(text))
        messagebox.showerror("음성 구간 확인", str(text), parent=self)

    def _segment(self, exclude=None):
        start, end = self._range()
        name = clip_filename(self.filename.get())
        for index, segment in enumerate(self.segments):
            if index == exclude:
                continue
            if (segment["start"], segment["end"]) == (start, end):
                raise ValueError("같은 구간이 이미 목록에 있습니다.")
            if segment["name"].casefold() == name.casefold():
                raise ValueError("목록의 구간마다 서로 다른 파일명을 입력하세요.")
        return {"start": start, "end": end, "name": name, "keep": self.disposition.get() == "저장"}

    def add_segment(self):
        if self.busy:
            return
        try:
            selected = self._segment()
        except ValueError as error:
            self._error(error)
            return
        self.segments.append(selected)
        self._refresh_segments()
        self.segment_list.see(str(len(self.segments) - 1))
        names = {segment["name"].casefold() for segment in self.segments}
        self.next_number += 1
        while default_clip_name(self.audio.source, self.next_number).casefold() in names:
            self.next_number += 1
        self.filename.set(default_clip_name(self.audio.source, self.next_number))
        self.disposition.set("저장")

    def update_segment(self):
        if self.busy:
            return
        selection = self.segment_list.selection()
        if len(selection) != 1:
            self._error("수정할 구간 하나를 목록에서 선택하세요.")
            return
        index = int(selection[0])
        try:
            selected = self._segment(exclude=index)
        except ValueError as error:
            self._error(error)
            return
        self.segments[index] = selected
        self._refresh_segments(index)

    def remove_segments(self):
        if not self.busy:
            for index in sorted((int(item) for item in self.segment_list.selection()), reverse=True):
                del self.segments[index]
            self._refresh_segments()

    def _refresh_segments(self, selected=None):
        self.segment_list.delete(*self.segment_list.get_children())
        for index, segment in enumerate(self.segments):
            start, end = segment["start"], segment["end"]
            self.segment_list.insert("", "end", iid=str(index), values=(segment["name"], f"{start:.3f}", f"{end:.3f}",
                f"{end - start:.3f}", "저장" if segment["keep"] else "제외"))
        if selected is not None:
            self.segment_list.selection_set(str(selected))
        count = sum(segment["keep"] for segment in self.segments)
        self.status.set(f"저장 {count}개 · 제외 {len(self.segments) - count}개 · 한 구간은 30초 이하입니다.")

    def _load_selected(self, _event=None):
        selection = self.segment_list.selection()
        if not self.busy and len(selection) == 1:
            segment = self.segments[int(selection[0])]
            self.start_var.set(str(segment["start"]))
            self.end_var.set(str(segment["end"]))
            self.filename.set(segment["name"])
            self.disposition.set("저장" if segment["keep"] else "제외")

    def stop_playback(self):
        try:
            self.stop_callback()
            return True
        except Exception as error:
            self._error(f"재생을 중지하지 못했습니다: {error}")
            return False

    def play_all(self):
        if not self.busy and self.stop_playback():
            self._play(self.audio.path)

    def _play(self, path):
        try:
            self.play_callback(path)
        except Exception as error:
            self._error(f"음성을 재생하지 못했습니다: {error}")

    def preview_selection(self):
        if self.busy:
            return
        try:
            selected = self._range()
        except ValueError as error:
            self._error(error)
            return
        self._run("preview", lambda: export_segments(self.audio, [selected], self.audio.path.parent / "preview"),
                  "선택 구간을 재생용 파일로 준비하고 있습니다…")

    def choose_output(self):
        if self.busy:
            return
        folder = filedialog.askdirectory(parent=self, title="구간 파일을 보관할 폴더", initialdir=str(Path(self.audio.source).parent), mustexist=False)
        if folder:
            self.output_dir.set(str(Path(folder).expanduser().resolve()))

    def save_segments(self):
        if self.busy:
            return
        selected = [segment for segment in self.segments if segment["keep"]]
        if not selected:
            self._error("저장할 구간이 없습니다. 구간을 담고 처리를 「저장」으로 지정하세요.")
            return
        folder = Path(self.output_dir.get()).expanduser().resolve()
        if folder.is_relative_to(self.audio.path.parent.resolve()):
            self._error("임시 음성 폴더 밖의 저장 폴더를 선택하세요. 임시 폴더는 편집기를 닫을 때 정리됩니다.")
            return
        ranges = [(segment["start"], segment["end"]) for segment in selected]
        names = [segment["name"] for segment in selected]
        self._run("save", lambda: export_segments(self.audio, ranges, folder, names=names),
                  "「저장」 구간만 각각 새 WAV 파일로 저장하고 있습니다…")

    def _set_busy(self, busy):
        self.busy = busy
        for widget in self.buttons + self.entries:
            widget.configure(state="disabled" if busy else "normal")
        self.disposition_input.configure(state="disabled" if busy else "readonly")

    def _run(self, kind, work, status):
        if not self.stop_playback():
            return
        self._set_busy(True)
        self.status.set(status)

        def run():
            try:
                self.events.put((kind, work(), None))
            except Exception as error:
                self.events.put((kind, None, str(error)))
        threading.Thread(target=run, daemon=True).start()

    def _poll(self):
        self.poll_job = None
        try:
            kind, result, error = self.events.get_nowait()
        except queue.Empty:
            pass
        else:
            self._set_busy(False)
            if error:
                if self.close_requested:
                    self._finish()
                else:
                    self._error(error)
            elif kind == "save":
                if self._finish():
                    self.on_saved(result)
            elif self.close_requested:
                self._finish()
            else:
                self._play(result[0])
                self.status.set("선택한 구간을 재생합니다.")
        if not self.closed:
            self.poll_job = self.after(50, self._poll)

    def close(self):
        if self.closed:
            return
        if self.busy:
            self.close_requested = True
            self.status.set("파일 작업이 끝나면 임시 음성을 정리하고 닫습니다…")
            return
        self._finish()

    def _finish(self):
        if self.closed:
            return False
        self.closed = True
        if self.poll_job is not None:
            self.after_cancel(self.poll_job)
            self.poll_job = None
        errors = []
        for action in (self.stop_callback, self.audio.close):
            try:
                action()
            except Exception as error:
                errors.append(str(error))
        try:
            self.grab_release()
        except tk.TclError:
            pass
        self.destroy()
        self.on_close()
        if errors:
            messagebox.showwarning("임시 음성 정리 확인", "\n".join(errors), parent=self.master)
        return True
