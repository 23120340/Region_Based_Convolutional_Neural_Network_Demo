from __future__ import annotations

import tkinter as tk
from collections.abc import Iterable
from tkinter import ttk

from .config import AssemblyConfig, load_config
from .fsm import ConfigurableAssemblyTracker, FsmOutcome
from .model_contract import Prediction
from .monitor import AssemblyMonitor, JsonlEventLogger
from .paths import DEFAULT_CONFIG, DEFAULT_EVENT_LOG
from .scenarios import SCENARIOS
from .smoother import TemporalDebouncer


RESULT_COLORS = {
    "PASS": "#16794b",
    "VIOLATION": "#b42318",
    "RESET": "#175cd3",
    "INFO": "#475467",
}


class PenAssemblyApp:
    def __init__(self, root: tk.Tk, config: AssemblyConfig) -> None:
        self.root = root
        self.config = config
        tracker = ConfigurableAssemblyTracker(config)
        debouncer = TemporalDebouncer(idle_actions=config.idle_actions)
        self.monitor = AssemblyMonitor(
            tracker,
            debouncer,
            JsonlEventLogger(DEFAULT_EVENT_LOG),
        )
        self._scenario_job: str | None = None

        root.title("Pen Assembly Monitor — MVP")
        root.geometry("1180x760")
        root.minsize(980, 650)
        root.configure(bg="#f2f4f7")

        self.state_var = tk.StringVar()
        self.status_var = tk.StringVar()
        self.expected_var = tk.StringVar()
        self.cycle_var = tk.StringVar()

        self._build_styles()
        self._build_layout()
        self._refresh(None)

    def _build_styles(self) -> None:
        style = ttk.Style()
        if "clam" in style.theme_names():
            style.theme_use("clam")
        style.configure("Card.TFrame", background="#ffffff")
        style.configure("Title.TLabel", background="#f2f4f7", foreground="#101828", font=("Segoe UI", 20, "bold"))
        style.configure("Subtitle.TLabel", background="#f2f4f7", foreground="#475467", font=("Segoe UI", 10))
        style.configure("CardTitle.TLabel", background="#ffffff", foreground="#101828", font=("Segoe UI", 12, "bold"))
        style.configure("Body.TLabel", background="#ffffff", foreground="#344054", font=("Segoe UI", 10))
        style.configure("State.TLabel", background="#ffffff", foreground="#175cd3", font=("Consolas", 17, "bold"))
        style.configure("Action.TButton", font=("Segoe UI", 10), padding=(10, 8))
        style.configure("Scenario.TButton", font=("Segoe UI", 9), padding=(8, 6))
        style.configure("Treeview", rowheight=28, font=("Segoe UI", 9))
        style.configure("Treeview.Heading", font=("Segoe UI", 9, "bold"))

    def _build_layout(self) -> None:
        outer = ttk.Frame(self.root, padding=20)
        outer.pack(fill="both", expand=True)

        ttk.Label(outer, text="Hệ thống giám sát lắp ráp bút bi", style="Title.TLabel").pack(anchor="w")
        ttk.Label(
            outer,
            text="MVP kiểm chứng debouncer + FSM; các nút đang mô phỏng đầu ra của model nhận diện.",
            style="Subtitle.TLabel",
        ).pack(anchor="w", pady=(2, 16))

        summary = ttk.Frame(outer, style="Card.TFrame", padding=16)
        summary.pack(fill="x")
        summary.columnconfigure(1, weight=1)
        ttk.Label(summary, text="TRẠNG THÁI", style="CardTitle.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(summary, textvariable=self.state_var, style="State.TLabel").grid(row=1, column=0, sticky="w", pady=(4, 0))
        ttk.Label(summary, textvariable=self.cycle_var, style="Body.TLabel").grid(row=2, column=0, sticky="w", pady=(4, 0))
        ttk.Label(summary, text="HƯỚNG DẪN / CẢNH BÁO", style="CardTitle.TLabel").grid(row=0, column=1, sticky="w", padx=(36, 0))
        self.status_label = tk.Label(
            summary,
            textvariable=self.status_var,
            bg="#ffffff",
            fg="#344054",
            font=("Segoe UI", 11, "bold"),
            anchor="w",
            justify="left",
            wraplength=650,
        )
        self.status_label.grid(row=1, column=1, rowspan=2, sticky="ew", padx=(36, 0), pady=(4, 0))

        body = ttk.Frame(outer)
        body.pack(fill="both", expand=True, pady=(14, 0))
        body.columnconfigure(0, weight=2)
        body.columnconfigure(1, weight=3)
        body.rowconfigure(0, weight=1)

        left = ttk.Frame(body, style="Card.TFrame", padding=16)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 7))
        right = ttk.Frame(body, style="Card.TFrame", padding=16)
        right.grid(row=0, column=1, sticky="nsew", padx=(7, 0))
        right.rowconfigure(1, weight=1)
        right.columnconfigure(0, weight=1)

        ttk.Label(left, text="1. Checklist quy trình", style="CardTitle.TLabel").pack(anchor="w")
        self.step_labels: dict[str, tk.Label] = {}
        for index, step in enumerate(self.config.workflow, start=1):
            label = tk.Label(
                left,
                text=f"○  {index}. {step.label}",
                bg="#ffffff",
                fg="#667085",
                font=("Segoe UI", 10),
                anchor="w",
                pady=5,
            )
            label.pack(fill="x")
            self.step_labels[step.action] = label

        ttk.Separator(left).pack(fill="x", pady=12)
        ttk.Label(left, text="2. Mô phỏng hành động nhận diện", style="CardTitle.TLabel").pack(anchor="w", pady=(0, 7))
        for step in self.config.workflow:
            ttk.Button(
                left,
                text=self.config.actions[step.action],
                style="Action.TButton",
                command=lambda action=step.action: self.submit_action(action),
            ).pack(fill="x", pady=3)

        ttk.Label(left, textvariable=self.expected_var, style="Body.TLabel", wraplength=390).pack(anchor="w", pady=(10, 4))
        ttk.Button(left, text="Đặt lại chu trình", command=self.reset).pack(fill="x", pady=(7, 0))

        ttk.Label(right, text="Kịch bản kiểm thử tự động", style="CardTitle.TLabel").grid(row=0, column=0, sticky="w")
        scenario_bar = ttk.Frame(right, style="Card.TFrame")
        scenario_bar.grid(row=0, column=0, sticky="e")
        scenario_labels = {
            "correct": "Đúng quy trình",
            "missing_spring": "Quên lò xo",
            "missing_refill": "Quên ruột",
            "premature_test": "Bấm thử sớm",
        }
        for name, label in scenario_labels.items():
            ttk.Button(
                scenario_bar,
                text=label,
                style="Scenario.TButton",
                command=lambda scenario=name: self.run_scenario(scenario),
            ).pack(side="left", padx=2)

        columns = ("type", "action", "transition", "message")
        self.events = ttk.Treeview(right, columns=columns, show="headings", height=13)
        self.events.heading("type", text="Kết quả")
        self.events.heading("action", text="Hành động")
        self.events.heading("transition", text="Chuyển trạng thái")
        self.events.heading("message", text="Thông báo")
        self.events.column("type", width=85, stretch=False)
        self.events.column("action", width=115, stretch=False)
        self.events.column("transition", width=200, stretch=False)
        self.events.column("message", width=360)
        self.events.grid(row=1, column=0, sticky="nsew", pady=(12, 0))
        scrollbar = ttk.Scrollbar(right, orient="vertical", command=self.events.yview)
        scrollbar.grid(row=1, column=1, sticky="ns", pady=(12, 0))
        self.events.configure(yscrollcommand=scrollbar.set)

    def submit_action(self, action: str) -> None:
        outcome = None
        # Five identical samples represent five consecutive recognizer windows.
        for _ in range(5):
            emitted = self.monitor.submit_prediction(Prediction(action, 0.95))
            if emitted is not None:
                outcome = emitted
        if outcome is not None:
            self._append_event(outcome)
            self._refresh(outcome)

    def run_scenario(self, name: str) -> None:
        if self._scenario_job is not None:
            self.root.after_cancel(self._scenario_job)
            self._scenario_job = None
        self.reset(cancel_scenario=False)
        self._run_actions(iter(SCENARIOS[name]))

    def _run_actions(self, actions: Iterable[str]) -> None:
        iterator = iter(actions)

        def advance() -> None:
            try:
                action = next(iterator)
            except StopIteration:
                self._scenario_job = None
                return
            self.submit_action(action)
            self._scenario_job = self.root.after(650, advance)

        advance()

    def reset(self, cancel_scenario: bool = True) -> None:
        if cancel_scenario and self._scenario_job is not None:
            self.root.after_cancel(self._scenario_job)
            self._scenario_job = None
        outcome = self.monitor.reset()
        self._append_event(outcome)
        self._refresh(outcome)

    def _append_event(self, outcome: FsmOutcome) -> None:
        transition = f"{outcome.previous_state} → {outcome.state}"
        self.events.insert("", 0, values=(outcome.type, outcome.action, transition, outcome.message))

    def _refresh(self, outcome: FsmOutcome | None) -> None:
        tracker = self.monitor.tracker
        self.state_var.set(tracker.state)
        self.cycle_var.set(f"Chu trình: {tracker.cycle_id or 'chưa bắt đầu'}")
        self.status_var.set(outcome.message if outcome else tracker.instruction)
        result_type = outcome.type if outcome else "INFO"
        self.status_label.configure(fg=RESULT_COLORS[result_type])

        expected_names = [self.config.actions[action] for action in tracker.expected_actions]
        self.expected_var.set("Bước hợp lệ tiếp theo: " + (", ".join(expected_names) or "không có"))
        completed = set(tracker.completed_steps)
        next_actions = set(tracker.expected_actions)
        for index, step in enumerate(self.config.workflow, start=1):
            label = self.step_labels[step.action]
            if step.action in completed:
                prefix, color, font = "✓", "#16794b", ("Segoe UI", 10, "bold")
            elif step.action in next_actions:
                prefix, color, font = "➜", "#b54708", ("Segoe UI", 10, "bold")
            else:
                prefix, color, font = "○", "#667085", ("Segoe UI", 10)
            label.configure(text=f"{prefix}  {index}. {step.label}", fg=color, font=font)


def run_app() -> None:
    config = load_config(DEFAULT_CONFIG)
    root = tk.Tk()
    PenAssemblyApp(root, config)
    root.mainloop()
