"""Local desktop workspace for reviewed audio pairs and Whisper fine-tuning."""
from datetime import datetime
import json
import os
from pathlib import Path
import queue
import shutil
import subprocess
import sys
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import uuid

from app_jobs import prepare_project, start_training, stop_training, training_dependencies
from review_store import import_recording, load_project, save_project


class RecordingRow:
    def __init__(self, app, record):
        self.app, self.record = app, record
        self.frame = ttk.Frame(app.row_area, padding=(10, 9), style="Row.TFrame")
        self.frame.pack(fill="x", pady=(0, 5))
        self.frame.columnconfigure(1, weight=1)
        left = ttk.Frame(self.frame, style="Row.TFrame", width=180)
        left.grid(row=0, column=0, sticky="nw", padx=(0, 12))
        ttk.Label(left, text=record.get("source_name", Path(record["audio"]).name),
                  wraplength=172, style="Row.TLabel").pack(anchor="w")
        ttk.Label(left, text=f"{record.get('duration', 0):.1f}초", style="MutedRow.TLabel").pack(anchor="w", pady=3)
        self.play = ttk.Button(left, text="▶ 다시 듣기", command=lambda: app.play(record))
        self.play.pack(anchor="w")
        self.text = tk.Text(self.frame, height=3, width=40, wrap="word", font=app.font,
                            relief="solid", borderwidth=1, padx=8, pady=5,
                            background="white", foreground="#172b4d", undo=True)
        self.text.grid(row=0, column=1, sticky="nsew", padx=(0, 12))
        self.text.insert("1.0", record.get("text", ""))
        self.text.edit_modified(False)
        self.text.bind("<<Modified>>", self._text_changed)
        right = ttk.Frame(self.frame, style="Row.TFrame")
        right.grid(row=0, column=2, sticky="nw")
        self.session = tk.StringVar(value=record["session"])
        self.language = tk.StringVar(value=record["language"])
        self.reviewed = tk.BooleanVar(value=record.get("reviewed") is True)
        ttk.Label(right, text="녹음 세션").grid(row=0, column=0, sticky="w")
        self.session_entry = ttk.Entry(right, textvariable=self.session, width=18)
        self.session_entry.grid(row=0, column=1, sticky="ew")
        self.language_label = ttk.Label(right, textvariable=self.language, width=5)
        self.language_label.grid(row=1, column=0, sticky="w", pady=5)
        self.check = ttk.Checkbutton(right, text="검수 완료", variable=self.reviewed, command=self._review_changed)
        self.check.grid(row=1, column=1, padx=(8, 0))
        self.hide = ttk.Button(right, text="화면에서 숨기기", command=lambda: app.hide_record(record["id"]))
        self.hide.grid(row=2, column=0, columnspan=2, sticky="ew")
        self.remove = ttk.Button(right, text="자료에서 제외", command=lambda: app.remove(record["id"]))
        self.remove.grid(row=3, column=0, columnspan=2, sticky="ew")
        self.session.trace_add("write", self._metadata_changed)
        self.language.trace_add("write", self._metadata_changed)

    def sync(self):
        self.record.update(text=self.text.get("1.0", "end-1c"), session=self.session.get(),
                           language=self.language.get(), reviewed=self.reviewed.get())
        self.text.edit_modified(False)

    def _text_changed(self, _event=None):
        if self.text.edit_modified():
            self.text.edit_modified(False)
            self.reviewed.set(False)
            self.sync()
            self.app.changed()

    def _metadata_changed(self, *_args):
        self.reviewed.set(False)
        self.sync()
        self.app.changed()

    def _review_changed(self):
        if self.reviewed.get() and not self.text.get("1.0", "end-1c").strip():
            self.reviewed.set(False)
            self.app.status.set("정답을 입력한 뒤 검수 완료를 선택하세요.")
        self.sync()
        self.app.changed()

    def set_enabled(self, enabled):
        state = "normal" if enabled else "disabled"
        for widget in (self.text, self.session_entry, self.play, self.check, self.hide, self.remove):
            widget.configure(state=state)


class ReviewApp:
    def __init__(self, root, project_dir=None):
        self.root = root
        self.font = ("맑은 고딕" if sys.platform == "win32" else "Noto Sans CJK KR", 10)
        self.project = Path(project_dir or Path(__file__).parent / "data" / "my-voice").resolve()
        self.records, self.views = [], []
        self.hidden_ids = set()
        self.events = queue.Queue()
        self.busy = self.dirty = self.close_after_stop = False
        self.closed = False
        self.process = self.player = self.run_output = None
        self.stop_requested = False
        self.actions = []
        self.project_path = tk.StringVar(value=str(self.project))
        self.session = tk.StringVar(value="recording-" + datetime.now().strftime("%Y%m%d-%H%M%S"))
        self.language = tk.StringVar(value="ko")
        self.common_answer = tk.StringVar()
        self.previous_data = tk.StringVar()
        self.adapter = tk.StringVar()
        self.device = tk.StringVar(value="자동")
        self.status = tk.StringVar(value="음성 파일과 정답을 같은 줄에서 관리합니다.")
        self.count = tk.StringVar(value="전체 0개 · 검수 완료 0개")
        self._build()
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.poll_job = self.root.after(80, self._poll)
        self.open_project(self.project, ask=False)

    def _button(self, parent, text, command, **kwargs):
        button = ttk.Button(parent, text=text, command=command, **kwargs)
        self.actions.append(button)
        return button

    def _build(self):
        root = self.root
        root.title("LViT 음성 학습 스튜디오")
        root.geometry("1180x850")
        root.minsize(980, 720)
        style = ttk.Style(root)
        style.theme_use("clam")
        style.configure(".", font=self.font)
        style.configure("TFrame", background="#f3f6fa")
        style.configure("TLabel", background="#f3f6fa", foreground="#172b4d")
        style.configure("Row.TFrame", background="white")
        style.configure("Row.TLabel", background="white", foreground="#172b4d")
        style.configure("MutedRow.TLabel", background="white", foreground="#68758b")
        style.configure("TButton", padding=(10, 6))
        style.configure("Accent.TButton", background="#147a78", foreground="white")
        style.map("Accent.TButton", background=[("active", "#106765"), ("disabled", "#b7c8cc")])
        style.configure("Title.TLabel", font=(self.font[0], 19, "bold"))
        outer = ttk.Frame(root, padding=18)
        outer.pack(fill="both", expand=True)
        outer.columnconfigure(0, weight=1)
        outer.rowconfigure(7, weight=1)
        ttk.Label(outer, text="LViT 음성 학습 스튜디오", style="Title.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(outer, text="음성 추가 → 정답 입력 → 검수 완료 → 일괄 저장 → 학습", foreground="#68758b").grid(row=1, column=0, sticky="w", pady=(3, 12))
        folder = ttk.Frame(outer)
        folder.grid(row=2, column=0, sticky="ew", pady=(0, 9))
        ttk.Label(folder, text="자료 폴더").pack(side="left", padx=(0, 8))
        ttk.Entry(folder, textvariable=self.project_path, state="readonly").pack(side="left", fill="x", expand=True)
        self._button(folder, "다른 폴더 열기", self.choose_project).pack(side="left", padx=(8, 0))
        tools = ttk.Frame(outer)
        tools.grid(row=3, column=0, sticky="ew", pady=(0, 5))
        self._button(tools, "+ 음성 파일 여러 개 추가", self.add_files, style="Accent.TButton").pack(side="left")
        ttk.Label(tools, text="새 파일의 녹음 세션").pack(side="left", padx=(16, 5))
        self.session_input = ttk.Entry(tools, textvariable=self.session, width=26)
        self.session_input.pack(side="left")
        ttk.Label(tools, text="새 파일 언어").pack(side="left", padx=(8, 0))
        self.language_input = ttk.Combobox(tools, textvariable=self.language, values=("ko", "en"), state="readonly", width=5)
        self.language_input.pack(side="left", padx=7)
        for event in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
            self.language_input.bind(event, self._language_wheel)
        self._button(tools, "재생 중지", self.stop_audio).pack(side="right")
        ttk.Label(outer, text="같은 원본의 필터별 녹음은 같은 세션으로 묶으세요. 학습·검증·평가에는 서로 다른 세션 3개 이상이 필요합니다.",
                  foreground="#68758b").grid(row=4, column=0, sticky="w", pady=(0, 8))
        common = ttk.Frame(outer)
        common.grid(row=5, column=0, sticky="ew", pady=(0, 10))
        ttk.Label(common, text="공통 정답").pack(side="left", padx=(0, 8))
        self.common_input = ttk.Entry(common, textvariable=self.common_answer)
        self.common_input.pack(side="left", fill="x", expand=True)
        self._button(common, "보이는 빈칸에 적용", self.apply_common).pack(side="left", padx=8)
        self._button(common, "보이는 항목 검수 완료", self.mark_all_reviewed).pack(side="left")
        heading = ttk.Frame(outer, padding=(10, 4))
        heading.grid(row=6, column=0, sticky="ew")
        ttk.Label(heading, text="음성 파일 · 재생", width=22).pack(side="left")
        ttk.Label(heading, text="정확하게 들은 정답 문장").pack(side="left", fill="x", expand=True)
        ttk.Label(heading, text="세션 · 언어 · 검수", width=23).pack(side="right")
        table = ttk.Frame(outer)
        table.grid(row=7, column=0, sticky="nsew")
        self.canvas = tk.Canvas(table, background="#e8edf4", highlightthickness=0)
        scrollbar = ttk.Scrollbar(table, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=scrollbar.set)
        self.canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        self.row_area = ttk.Frame(self.canvas)
        self.canvas_window = self.canvas.create_window((0, 0), window=self.row_area, anchor="nw")
        self.row_area.bind("<Configure>", lambda _event: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.canvas.bind("<Configure>", lambda event: self.canvas.itemconfigure(self.canvas_window, width=event.width))
        root.bind_all("<MouseWheel>", self._wheel)
        root.bind_all("<Button-4>", self._wheel)
        root.bind_all("<Button-5>", self._wheel)
        footer = ttk.Frame(outer)
        footer.grid(row=8, column=0, sticky="ew", pady=10)
        ttk.Label(footer, textvariable=self.count).pack(anchor="w")
        visibility = ttk.Frame(footer)
        visibility.pack(fill="x", pady=(5, 0))
        self._button(visibility, "현재 목록 숨기기", self.hide_visible).pack(side="left")
        self._button(visibility, "전체 다시 보기", self.show_all).pack(side="left", padx=5)
        self._button(visibility, "보이는 항목 언어 적용", self.apply_language).pack(side="left")
        self._button(visibility, "초안·검수 자료 일괄 저장", self.save, style="Accent.TButton").pack(side="right")
        training = ttk.LabelFrame(outer, text="학습 실행", padding=9)
        training.grid(row=9, column=0, sticky="ew")
        options = ttk.Frame(training)
        options.pack(fill="x")
        ttk.Label(options, text="장치").pack(side="left")
        self.device_input = ttk.Combobox(options, textvariable=self.device, values=("자동", "CPU", "GPU"), state="readonly", width=7)
        self.device_input.pack(side="left", padx=(5, 10))
        self._button(options, "1단계 학습 점검", lambda: self.train(quick=True)).pack(side="left", padx=(0, 5))
        self._button(options, "학습 시작", self.train, style="Accent.TButton").pack(side="left")
        self.stop_button = ttk.Button(options, text="학습 중지", command=self.stop, state="disabled")
        self.stop_button.pack(side="left", padx=5)
        self._button(options, "이전 모델 선택", self.choose_adapter).pack(side="right")
        self._button(options, "기존 분할 선택", self.choose_previous).pack(side="right", padx=5)
        ttk.Label(training, textvariable=self.adapter, foreground="#68758b", wraplength=1080).pack(anchor="w", pady=(5, 0))
        self.log = tk.Text(training, height=5, wrap="word", font=("Consolas", 9), state="disabled",
                           background="#172b4d", foreground="#edf5ff", relief="flat", padx=8, pady=6)
        self.log.pack(fill="x", pady=(5, 0))
        self.status_label = ttk.Label(outer, textvariable=self.status, foreground="#146c6b", wraplength=1100)
        self.status_label.grid(row=10, column=0, sticky="ew", pady=(8, 0))
        outer.bind("<Configure>", lambda event: self.status_label.configure(wraplength=max(280, event.width - 20)))

    def _wheel(self, event):
        if isinstance(event.widget, tk.Text):
            return
        direction = -1 if getattr(event, "num", None) == 4 else 1
        if getattr(event, "delta", 0):
            direction = -1 if event.delta > 0 else 1
        self.canvas.yview_scroll(direction * 3, "units")

    def _language_wheel(self, event):
        self._wheel(event)
        return "break"

    def _sync(self):
        for view in self.views:
            view.sync()

    def changed(self):
        self.dirty = True
        self._refresh_count()

    def _refresh_count(self):
        ready = sum(row.get("reviewed") is True for row in self.records)
        hidden = sum(row["id"] in self.hidden_ids for row in self.records)
        sessions = len({row["session"].strip() for row in self.records if row.get("reviewed") and row["session"].strip()})
        self.count.set(f"전체 {len(self.records)}개 · 화면 {len(self.records) - hidden}개 · 숨김 {hidden}개 · 검수 완료 {ready}개 · 초안 {len(self.records) - ready}개 · 검수 세션 {sessions}개")

    def render(self):
        for child in self.row_area.winfo_children():
            child.destroy()
        self.views = [RecordingRow(self, record) for record in self.records if record["id"] not in self.hidden_ids]
        if not self.records:
            ttk.Label(self.row_area, text="음성 파일을 여러 개 추가하세요.\n각 파일 오른쪽에 정답을 입력할 수 있습니다.",
                      justify="center", padding=35).pack(fill="x")
        elif not self.views:
            ttk.Label(self.row_area, text=f"자료 {len(self.records)}개를 화면에서 숨겼습니다. 모두 저장되며 검수 완료 자료는 학습에 포함됩니다.\n「전체 다시 보기」로 다시 표시할 수 있습니다.",
                      justify="center", padding=35).pack(fill="x")
        self._refresh_count()

    def hide_record(self, identity):
        self._sync()
        self.hidden_ids.add(identity)
        self.render()
        self.status.set("화면에서만 숨겼습니다. 자료는 유지되며 「전체 다시 보기」로 확인할 수 있습니다.")

    def hide_visible(self):
        self._sync()
        self.hidden_ids.update(view.record["id"] for view in self.views)
        self.render()
        self.status.set("현재 목록을 화면에서만 숨겼습니다. 기존 자료도 함께 저장되며 검수 완료 자료는 학습에 포함됩니다.")

    def show_all(self):
        self._sync()
        self.hidden_ids.clear()
        self.render()
        self.status.set("숨긴 자료를 모두 다시 표시했습니다.")

    def _busy(self, enabled, status=None):
        self.busy = enabled
        if enabled:
            self.stop_audio()
        state = "disabled" if enabled else "normal"
        for button in self.actions:
            button.configure(state=state)
        for widget in (self.session_input, self.common_input):
            widget.configure(state=state)
        for widget in (self.language_input, self.device_input):
            widget.configure(state="disabled" if enabled else "readonly")
        for view in self.views:
            view.set_enabled(not enabled)
        self.stop_button.configure(state="normal" if self.process else "disabled")
        if status:
            self.status.set(status)

    def _job(self, work, finished, status):
        self._busy(True, status)

        def run():
            try:
                self.events.put(("job", finished, work(), None))
            except Exception as error:
                self.events.put(("job", finished, None, str(error)))
        threading.Thread(target=run, daemon=True).start()

    def _poll(self):
        for _ in range(150):
            try:
                event = self.events.get_nowait()
            except queue.Empty:
                break
            if event[0] == "job":
                _, finished, result, error = event
                self._busy(False)
                if error:
                    self.status.set(error)
                    messagebox.showerror("작업을 완료하지 못했습니다", error, parent=self.root)
                else:
                    finished(result)
            elif event[0] == "log":
                self._log(event[1])
            elif event[0] == "exit":
                self._training_done(event[1])
            elif event[0] == "stop_error":
                self._log(event[1])
                if self.process and self.process.poll() is None:
                    self.stop_requested = False
                    self.close_after_stop = False
                    self.stop_button.configure(state="normal")
                    self.status.set("학습을 중지하지 못했습니다. 로그를 확인한 뒤 다시 시도하세요.")
            if self.closed:
                return
        if not self.closed:
            self.poll_job = self.root.after(80, self._poll)

    def open_project(self, path, ask=True):
        if ask and self.dirty and not messagebox.askyesno("미저장 변경", "저장하지 않은 변경을 버리고 다른 자료 폴더를 열까요?", parent=self.root):
            return
        path = Path(path).expanduser().resolve()

        def loaded(rows):
            self.stop_audio()
            self.project, self.records = path, rows
            self.hidden_ids.clear()
            self.project_path.set(str(path))
            self.previous_data.set("")
            self.adapter.set("")
            state = path / ".lvit-app.json"
            if state.exists():
                try:
                    values = json.loads(state.read_text(encoding="utf-8"))
                    self.previous_data.set(values.get("previous_data", ""))
                    self.adapter.set(values.get("adapter", ""))
                except (ValueError, OSError, AttributeError) as error:
                    messagebox.showwarning("학습 기록 확인", f"학습 연결 기록을 읽을 수 없습니다: {error}", parent=self.root)
            self.render()
            self.dirty = False
            self.status.set("자료를 불러왔습니다. 미검수 항목은 초안으로 보관됩니다.")
        self._job(lambda: load_project(path), loaded, "자료 폴더를 확인하고 있습니다…")

    def choose_project(self):
        path = filedialog.askdirectory(title="음성·정답을 보관할 자료 폴더", parent=self.root)
        if path:
            self.open_project(path)

    def add_files(self):
        if not self.session.get().strip():
            messagebox.showwarning("녹음 세션", "새 파일의 녹음 세션을 입력하세요.", parent=self.root)
            return
        paths = filedialog.askopenfilenames(parent=self.root, title="음성 파일 여러 개 선택",
            filetypes=(("음성 파일", "*.wav *.flac *.ogg *.mp3 *.m4a *.aac *.opus *.wma *.aiff"), ("모든 파일", "*.*")))
        if not paths:
            return
        self._sync()
        session, language = self.session.get().strip(), self.language.get()
        existing = {row["id"] for row in self.records}

        def work():
            imported, errors, seen = [], [], set(existing)
            for path in paths:
                try:
                    row = import_recording(path, self.project, session, language)
                    if row["id"] in seen:
                        raise ValueError("같은 음성이 이미 등록되어 있습니다. 숨긴 자료는 「전체 다시 보기」로 확인하세요.")
                    seen.add(row["id"])
                    imported.append(row)
                except Exception as error:
                    errors.append(f"{Path(path).name}: {error}")
            return imported, errors

        def added(result):
            imported, errors = result
            self.records.extend(imported)
            if imported:
                self.changed()
            self.render()
            self.status.set(f"{len(imported)}개 추가 · {len(errors)}개 제외. 정답을 입력한 뒤 검수 완료를 선택하세요.")
            if errors:
                messagebox.showwarning("추가하지 못한 파일", "\n".join(errors[:8]), parent=self.root)
        self._job(work, added, "음성을 검사하고 재생 가능한 WAV로 준비하고 있습니다…")

    def remove(self, identity):
        self._sync()
        if not messagebox.askyesno("학습 자료에서 제외", "이 항목의 정답·세션·검수 기록을 학습 자료에서 제외할까요?\n일괄 저장하면 저장된 목록에도 반영됩니다. 음성 파일 자체는 유지됩니다.\n\n화면만 정리하려면 취소 후 「화면에서 숨기기」를 사용하세요.", parent=self.root):
            return
        self.records = [row for row in self.records if row["id"] != identity]
        self.hidden_ids.discard(identity)
        self.changed()
        self.render()
        self.status.set("자료에서 제외했습니다. 일괄 저장하면 저장된 목록에도 반영됩니다. 음성 파일은 유지됩니다.")

    def apply_common(self):
        text = self.common_answer.get().strip()
        if not text:
            self.status.set("공통 정답을 먼저 입력하세요.")
            return
        count = 0
        for view in self.views:
            if not view.text.get("1.0", "end-1c").strip():
                view.text.insert("1.0", text)
                view.reviewed.set(False)
                view.sync()
                count += 1
        self.changed()
        self.status.set(f"화면에 보이는 빈 정답칸 {count}개에만 적용했습니다. 재생해서 확인한 뒤 검수 완료를 선택하세요.")

    def mark_all_reviewed(self):
        self._sync()
        for view in self.views:
            view.reviewed.set(bool(view.record["text"].strip() and view.record["session"].strip()))
            view.sync()
        self.changed()

    def apply_language(self):
        self._sync()
        language = self.language.get()
        count = 0
        for view in self.views:
            if view.language.get() != language:
                view.language.set(language)
                count += 1
        self.status.set(f"보이는 항목 {count}개의 언어를 {language}로 수정했습니다. 변경한 항목은 다시 검수 완료한 뒤 저장하세요.")

    def save(self, finished=None):
        self._sync()
        snapshot = [dict(row) for row in self.records]

        def saved(result):
            path, count = result
            self.dirty = False
            self.status.set(f"저장 완료 · 숨긴 자료 포함 학습 자료 {count}개 · 미검수 초안 {len(snapshot) - count}개 · {path}")
            if finished:
                finished()
        self._job(lambda: save_project(self.project, snapshot), saved, "초안과 검수 완료 자료를 저장하고 있습니다…")

    def play(self, record):
        path = self.project / record["audio"]
        try:
            self.stop_audio()
            if sys.platform == "win32":
                import winsound
                winsound.PlaySound(str(path), winsound.SND_FILENAME | winsound.SND_ASYNC)
            else:
                command = shutil.which("afplay" if sys.platform == "darwin" else "ffplay")
                if not command:
                    raise ValueError("이 환경에는 재생기가 없습니다. Windows에서는 기본 재생을 지원합니다.")
                args = [command, str(path)] if sys.platform == "darwin" else [command, "-nodisp", "-autoexit", "-loglevel", "quiet", str(path)]
                self.player = subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            self.status.set(f"재생 중: {record.get('source_name', path.name)}")
        except (OSError, ValueError, RuntimeError) as error:
            messagebox.showerror("재생 실패", str(error), parent=self.root)

    def stop_audio(self):
        if sys.platform == "win32":
            import winsound
            winsound.PlaySound(None, 0)
        if self.player and self.player.poll() is None:
            self.player.terminate()
        self.player = None

    def choose_adapter(self):
        path = filedialog.askdirectory(title="이전 학습의 best 폴더 (취소하면 현재 설정 유지)", parent=self.root)
        if path:
            self.adapter.set(path)
        elif self.adapter.get() and messagebox.askyesno("기본 모델에서 시작", "선택한 이전 모델을 해제할까요?", parent=self.root):
            self.adapter.set("")

    def choose_previous(self):
        path = filedialog.askdirectory(title="기존 train/validation/test가 있는 준비 폴더", parent=self.root)
        if path:
            self.previous_data.set(path)
            self.status.set(f"기존 분할 유지: {path}")

    def train(self, quick=False):
        self._sync()
        if not any(row.get("reviewed") is True for row in self.records):
            messagebox.showwarning("학습 자료", "정답을 입력하고 검수 완료를 선택한 자료가 필요합니다.", parent=self.root)
            return
        missing = training_dependencies()
        if missing:
            messagebox.showinfo("학습 환경 설치 필요", "자료 입력·저장은 지금 사용할 수 있습니다.\nREADME.md의 '명령행 학습 환경 설치' 안내에 따라 학습 패키지를 설치하세요.\n\n필요한 항목: " + ", ".join(missing), parent=self.root)
            return
        snapshot = [dict(row) for row in self.records]
        previous, adapter = self.previous_data.get() or None, self.adapter.get() or None
        device = {"자동": "auto", "CPU": "cpu", "GPU": "cuda"}[self.device.get()]

        def work():
            save_project(self.project, snapshot)
            prepared = prepare_project(self.project, previous)
            process, output = start_training(self.project, prepared, device, adapter, quick)
            return process, output, prepared

        def started(result):
            self.process, self.run_output, prepared = result
            self.previous_data.set(str(prepared))
            self.stop_requested = False
            self.dirty = False
            self._write_state()
            self._busy(True, "학습 중입니다. 첫 실행에는 기본 모델을 다운로드합니다.")
            self._log(f"\n학습 시작 · 결과: {self.run_output}\n")
            process = self.process

            def read_log():
                try:
                    for line in process.stdout:
                        self.events.put(("log", line))
                except (OSError, ValueError):
                    pass
                finally:
                    self.events.put(("exit", process.wait()))
            threading.Thread(target=read_log, daemon=True).start()
        self._job(work, started, "저장한 정답으로 학습 자료를 새로 준비하고 있습니다…")

    def _log(self, text):
        self.log.configure(state="normal")
        self.log.insert("end", text)
        if int(self.log.index("end-1c").split(".")[0]) > 600:
            self.log.delete("1.0", "150.0")
        self.log.see("end")
        self.log.configure(state="disabled")

    def stop(self):
        if not self.process:
            return
        self.stop_requested = True
        self.stop_button.configure(state="disabled")
        self.status.set("학습 중지를 요청했습니다…")
        process = self.process

        def work():
            try:
                stop_training(process)
            except Exception as error:
                self.events.put(("stop_error", str(error)))
        threading.Thread(target=work, daemon=True).start()

    def _training_done(self, code):
        process = self.process
        self.process = None
        if process and process.stdout and not process.stdout.closed:
            process.stdout.close()
        self._busy(False)
        if self.stop_requested:
            self.status.set("학습을 중지했습니다. 저장된 녹음과 정답은 유지됩니다.")
        elif code == 0:
            self.adapter.set(str(self.run_output / "best"))
            self._write_state()
            self.status.set(f"학습 완료 · {self.run_output / 'best'} · 정확도는 별도 평가로 확인하세요.")
        else:
            self.status.set("학습이 종료되었습니다. 로그의 오류를 확인하세요. GPU 메모리가 부족하면 cpu 장치로 점검할 수 있습니다.")
        if self.close_after_stop:
            self.destroy()

    def _write_state(self):
        try:
            self.project.mkdir(parents=True, exist_ok=True)
            temporary = self.project / (".lvit-app-" + uuid.uuid4().hex + ".tmp")
            temporary.write_text(json.dumps({"previous_data": self.previous_data.get(), "adapter": self.adapter.get()}, ensure_ascii=False), encoding="utf-8")
            os.replace(temporary, self.project / ".lvit-app.json")
        except OSError as error:
            self._log(f"학습 연결 기록을 저장하지 못했습니다: {error}\n")

    def close(self):
        if self.process:
            if messagebox.askyesno("학습 중", "학습을 중지하고 앱을 닫을까요?", parent=self.root):
                self.close_after_stop = True
                self.stop()
            return
        if self.busy:
            messagebox.showinfo("처리 중", "파일 처리가 끝난 뒤 앱을 닫아주세요.", parent=self.root)
            return
        if self.dirty:
            answer = messagebox.askyesnocancel("저장하지 않은 변경", "초안과 검수 자료를 저장하고 닫을까요?", parent=self.root)
            if answer is None:
                return
            if answer:
                self.save(finished=self.destroy)
                return
        self.destroy()

    def destroy(self):
        self.closed = True
        self.root.after_cancel(self.poll_job)
        self.stop_audio()
        self.root.destroy()


def main():
    root = tk.Tk()
    ReviewApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
