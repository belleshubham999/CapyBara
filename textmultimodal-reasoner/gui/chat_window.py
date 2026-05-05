"""Modern chat UI using CustomTkinter + TkinterDnD2."""

from __future__ import annotations

import threading
import time
from pathlib import Path
from tkinter import filedialog, messagebox

import customtkinter as ctk
from PIL import Image
from tkinterdnd2 import DND_FILES, TkinterDnD

from core.model_engine import ModelEngine
from gui.drag_drop import normalize_drop_path
from utils.ram_cleaner import run_clean
from utils.logger import append_hf_example


class ChatWindow(TkinterDnD.Tk):
    def __init__(self, model_engine: ModelEngine) -> None:
        super().__init__()
        self.engine = model_engine
        self.title("TextMultimodal Reasoner")
        self.geometry("980x700")

        self.current_attachment: Path | None = None
        self.thumbnail_img: ctk.CTkImage | None = None
        self.system_prompt = ctk.StringVar(value="You are a helpful multimodal reasoning assistant.")

        self._closing = threading.Event()
        self.ramclear_call_count = 0

        self._build_layout()
        self._show_runtime_modes()
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        threading.Thread(target=self._ramclear_loop, daemon=True).start()

    def _ramclear_loop(self) -> None:
        while not self._closing.is_set():
            try:
                run_clean(protected_pids={})
                self.ramclear_call_count += 1
            except Exception:
                pass
            self._closing.wait(30.0)

    def _build_layout(self) -> None:
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)
        header = ctk.CTkFrame(self)
        header.grid(row=0, column=0, sticky="ew", padx=12, pady=(12, 6))
        settings_btn = ctk.CTkButton(header, text="Settings", width=120, command=self._open_settings)
        settings_btn.pack(side="right", padx=6, pady=8)
        self.mode_label = ctk.CTkLabel(header, text="")
        self.mode_label.pack(side="left", padx=8)
        self.chat_history = ctk.CTkTextbox(self, wrap="word")
        self.chat_history.grid(row=1, column=0, sticky="nsew", padx=12, pady=6)
        self.chat_history.configure(state="disabled")
        bottom = ctk.CTkFrame(self)
        bottom.grid(row=2, column=0, sticky="ew", padx=12, pady=(6, 12))
        bottom.grid_columnconfigure(0, weight=1)
        self.preview_label = ctk.CTkLabel(bottom, text="Drop image/audio here or use Attach")
        self.preview_label.grid(row=0, column=0, sticky="w", padx=8, pady=(8, 2))
        input_row = ctk.CTkFrame(bottom)
        input_row.grid(row=1, column=0, sticky="ew", padx=8, pady=8)
        input_row.grid_columnconfigure(0, weight=1)
        self.input_box = ctk.CTkEntry(input_row, placeholder_text="Ask something...")
        self.input_box.grid(row=0, column=0, sticky="ew", padx=(0, 6), pady=6)
        ctk.CTkButton(input_row, text="Attach", width=90, command=self._pick_file).grid(row=0, column=1, padx=4)
        ctk.CTkButton(input_row, text="Send", width=90, command=self._send_message).grid(row=0, column=2, padx=(4, 0))
        self.drop_target_register(DND_FILES)
        self.dnd_bind("<<Drop>>", self._on_drop)

    def _show_runtime_modes(self) -> None:
        self.mode_label.configure(text="KV cache: quantized (forced) | mmap: ON (forced)")

    def _open_settings(self) -> None:
        win = ctk.CTkToplevel(self)
        win.title("Settings")
        win.geometry("520x180")
        ctk.CTkLabel(win, text="System Prompt").pack(anchor="w", padx=12, pady=(12, 4))
        prompt_box = ctk.CTkTextbox(win, height=90)
        prompt_box.pack(fill="both", expand=True, padx=12, pady=4)
        prompt_box.insert("1.0", self.system_prompt.get())

        def save_prompt() -> None:
            self.system_prompt.set(prompt_box.get("1.0", "end").strip())
            win.destroy()

        ctk.CTkButton(win, text="Save", command=save_prompt).pack(pady=10)

    def _pick_file(self) -> None:
        file_path = filedialog.askopenfilename()
        if file_path:
            self._set_attachment(Path(file_path))

    def _on_drop(self, event) -> None:  # noqa: ANN001
        try:
            self._set_attachment(normalize_drop_path(event.data))
        except Exception as exc:
            messagebox.showerror("Drop Error", str(exc))

    def _set_attachment(self, path: Path) -> None:
        if not path.exists():
            messagebox.showerror("Attachment", f"File not found: {path}")
            return
        self.current_attachment = path
        if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".webp"}:
            img = Image.open(path)
            img.thumbnail((96, 96))
            self.thumbnail_img = ctk.CTkImage(light_image=img, dark_image=img, size=img.size)
            self.preview_label.configure(text=path.name, image=self.thumbnail_img, compound="left")
        else:
            self.preview_label.configure(text=f"Attached: {path.name}", image=None)

    def _append_chat(self, title: str, body: str) -> None:
        self.chat_history.configure(state="normal")
        self.chat_history.insert("end", f"\n[{title}]\n{body}\n")
        self.chat_history.see("end")
        self.chat_history.configure(state="disabled")

    def _send_message(self) -> None:
        user_text = self.input_box.get().strip()
        if not user_text and not self.current_attachment:
            return
        self.input_box.delete(0, "end")
        self._append_chat("User", user_text or "(attachment only)")
        payload = []
        if self.current_attachment:
            ext = self.current_attachment.suffix.lower()
            payload.append({"type": "image" if ext in {'.png','.jpg','.jpeg','.bmp','.webp'} else "audio", "data": str(self.current_attachment)})
        payload.append({"type": "text", "data": f"{self.system_prompt.get()}\n\n{user_text}"})
        self._append_chat("Reasoning", "...thinking...")
        threading.Thread(target=self._run_inference, args=(payload,), daemon=True).start()

    def _run_inference(self, payload: list[dict[str, str]]) -> None:
        reasoning_chunks: list[str] = []

        def on_reasoning(token: str) -> None:
            reasoning_chunks.append(token)

        answer = self.engine.generate_response(payload, on_reasoning_token=on_reasoning)

        reasoning_text = "".join(reasoning_chunks).strip()

        def finish_ui() -> None:
            if reasoning_text:
                self._append_chat("Reasoning", reasoning_text)
            self._append_chat("Final Answer", answer)

            user_question = ""
            for item in payload:
                if item.get("type") == "text":
                    user_question = item.get("data", "")
            append_hf_example(
                "logs/hf_training_data.txt",
                question=user_question,
                reasoning=reasoning_text,
                answer=answer,
                metadata={"attachment": str(self.current_attachment) if self.current_attachment else ""},
            )

        self.after(0, finish_ui)

    def _on_close(self) -> None:
        self._closing.set()
        self.engine.unload()
        print(f"ram_cleaner calls this session: {self.ramclear_call_count}")
        self.destroy()


def run_chat_app(model_engine: ModelEngine) -> int:
    app = ChatWindow(model_engine)
    app.mainloop()
    return app.ramclear_call_count
