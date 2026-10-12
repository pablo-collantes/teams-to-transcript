"""
Companion engine - a GUI-agnostic local-LLM meeting advisor embedded in the app.

It:

  * spawns (or reuses) a llama.cpp server for a local chat model (any GGUF chat
    model; OpenAI-compatible API),
  * loads a static context pack (every .txt/.md file in `companion_context_dir`,
    e.g. an agenda, a spec, notes from the last meeting) as the system prompt once,
    so llama's prefix KV-cache is reused each turn,
  * keeps the live meeting transcript append-only (fed in-process by the capture
    manager - no HTTP polling needed since we run inside the same app),
  * answers on demand (a typed question / "talking points" button) and, optionally,
    drops a brief ambient note when the speaker pauses.

Everything is standard library. No PyQt import here on purpose: the engine talks to
the UI only through a single `emit(kind, payload)` callback, so its background
threads can be marshalled onto the GUI thread by the caller (via Qt signals).

emit() event kinds:
    ("status", str)            engine lifecycle / info line
    ("ready", bool)            True once the model is warmed and responsive
    ("response_start", str)    a new answer is starting; payload is its header
    ("token", str)             a streamed piece of the current answer
    ("response_end", None)     the current answer finished
    ("error", str)             a non-fatal error message
"""
from __future__ import annotations

import html
import json
import os
import shutil
import subprocess
import tempfile
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Callable, List, Optional


# Files in companion_context_dir that are loaded as reference documents.
CONTEXT_SUFFIXES = (".txt", ".md")

# --- instructions ----------------------------------------------------------
AMBIENT_INSTR = (
    "The speaker just paused. Looking especially at the most recent lines, drop a brief "
    "note (at most 3 bullets) on the single most useful thing for me to say or watch on "
    "the current point. If there is nothing actionable, reply with exactly '—'."
)
CURRENT_POINT_INSTR = (
    "Give me talking points on what is being discussed right now (the most recent lines): "
    "the strongest thing to say, which reference document it touches (if any), and a WARN "
    "if someone is about to agree to something we should not or says something the "
    "reference material contradicts. Follow the RULES."
)


def _question_instr(q: str) -> str:
    return (
        f"I typed this question during the meeting: \"{q}\"\n"
        "Answer concisely and specifically, grounded in the reference documents where they "
        "are relevant, citing them by source id. Use the transcript above for context "
        "if the question refers to what was just discussed. Up to ~6 bullets."
    )


# Built-in generic advisor preamble. Overridable by dropping a `system.txt` into
# the context dir (its contents replace this block verbatim) - that is the place to
# say who you are, what the meeting is for and what to watch out for.
_DEFAULT_PREAMBLE = """You are a silent, real-time advisor sitting in on a live meeting. The \
meeting transcript streams in as people speak, and any reference documents the user prepared \
(agenda, notes, specs, prior decisions) are included below.

Your job: give the user instant, glanceable, meeting-ready guidance. The user is talking to \
other people and can only glance at your output, so be terse and high-signal.

RULES:
- Be terse: usually 2-4 short bullets (fewer for a brief ambient note, a few more only \
when answering a typed question - the instruction at the end of each turn says which). No \
preamble, no restating what was said, no sign-off.
- Prioritize, in order: (a) a concrete point to make or question to ask about what was just \
said, grounded in the reference documents when they cover it; (b) which document / section \
it relates to; (c) a warning, prefixed "WARN:", if someone states something the documents \
contradict, or the user is about to commit to something they may not want to.
- If the latest speech is small talk, logistics, or nothing actionable, reply with exactly "—" \
(one em dash) and nothing else.
- Never invent facts, numbers, or quotes that are not in the transcript or the documents. If \
you are unsure, say so in a few words."""


class CompanionConfig:
    """Reads companion settings from the app's settings dict."""

    def __init__(self, settings):
        # `.get(key, default)` returns "" when the key exists but is blank, so fall
        # back with `or` to keep empty settings meaning "use the default".
        self.llama_bin = os.path.expanduser(
            settings.get("companion_llama_bin", "") or shutil.which("llama-server") or "")
        self.model = os.path.expanduser(settings.get("companion_model", "") or "")
        self.host = settings.get("companion_host", "127.0.0.1")
        self.port = int(settings.get("companion_port", 8090))
        self.ctx = int(settings.get("companion_ctx", 32768))
        self.ngl = int(settings.get("companion_ngl", 999))
        self.threads = int(settings.get("companion_threads", 8))
        self.ready_timeout = float(settings.get("companion_ready_timeout", 240))

        self.max_tokens = int(settings.get("companion_max_tokens", 320))
        self.temperature = float(settings.get("companion_temperature", 0.3))

        context_dir = settings.get("companion_context_dir", "")
        self.context_dir = Path(context_dir).expanduser() if context_dir else None
        self.max_transcript_chars = int(settings.get("companion_max_transcript_chars", 20000))

        self.ambient_default = bool(settings.get("companion_ambient", True))
        self.pause_secs = float(settings.get("companion_pause_secs", 7))
        self.ambient_min_new_chars = int(settings.get("companion_ambient_min_chars", 140))
        self.ambient_cooldown = float(settings.get("companion_ambient_cooldown", 12))

        self.log_path = os.path.expanduser(
            settings.get("companion_log", "")
            or os.path.join(tempfile.gettempdir(), "teams-to-transcript-llama-server.log"))

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"


class CompanionEngine:
    """Local-LLM advisor. Thread-safe; drives the UI through `emit(kind, payload)`."""

    def __init__(self, config: CompanionConfig,
                 emit: Callable[[str, object], None]):
        self.cfg = config
        self._emit = emit
        self.system_prompt = ""
        self.doc_count = 0

        self._proc: Optional[subprocess.Popen] = None
        self._spawned = False           # did we start llama-server (vs reuse)?
        self.ready = False
        self._stopping = False

        # transcript state (append-only for KV-cache reuse)
        self._lock = threading.Lock()
        self._lines: List[str] = []
        self._responded_len = 0
        self._last_speech_mono = 0.0
        self._last_response_mono = 0.0
        self.ambient_enabled = self.cfg.ambient_default

        # one generation at a time (llama serves one request; concurrent calls
        # would thrash the KV cache and interleave output)
        self._gen_lock = threading.Lock()
        self._ambient_thread: Optional[threading.Thread] = None

    # -- small helpers -------------------------------------------------
    def emit(self, kind: str, payload=None):
        try:
            self._emit(kind, payload)
        except Exception:
            pass

    def _status(self, msg: str):
        self.emit("status", msg)

    # -- system prompt / context --------------------------------------
    def build_system_prompt(self) -> str:
        cd = self.cfg.context_dir
        docs: List[Path] = []
        override: Optional[Path] = None
        if cd is not None:
            if cd.is_dir():
                override = cd / "system.txt"
                docs = sorted(p for p in cd.iterdir()
                              if p.is_file() and p.suffix.lower() in CONTEXT_SUFFIXES
                              and p.name != "system.txt")
            else:
                self.emit("error", f"companion_context_dir not found: {cd}")

        preamble = (override.read_text(encoding="utf-8", errors="replace")
                    if override is not None and override.exists() else _DEFAULT_PREAMBLE)
        self.doc_count = len(docs)

        if not docs:
            return f"""{preamble}

No reference documents were provided: work from the transcript alone, and say so in a few \
words when a point would need information you do not have."""

        # Wrap each document in an id-tagged <source> block and tell the model to
        # ground + cite against them. This is static across turns, so llama's prefix
        # KV-cache still covers the whole context pack.
        sources = "\n\n".join(
            f'<source id="{html.escape(p.stem)}" title="{html.escape(p.name)}">\n'
            f'{p.read_text(encoding="utf-8", errors="replace")}\n'
            '</source>'
            for p in docs)
        return f"""{preamble}

--- HOW TO USE THE SOURCES BELOW ---
Everything after this line is reference material, each document wrapped in a <source id="..."> tag.
Ground claims, numbers, and quotes in these sources or the transcript - never state a figure or
quote that is in neither. Cite a document inline by its source id, plus a section / figure / table
number when it has one (e.g. "(spec §3.2)").
If the sources do not settle a point, say so in a few words rather than inventing an answer.

{sources}
"""

    # -- llama.cpp server lifecycle -----------------------------------
    def _healthy(self) -> bool:
        try:
            with urllib.request.urlopen(f"{self.cfg.base_url}/health", timeout=2.0) as r:
                return r.status == 200
        except Exception:
            return False

    def start(self):
        """Non-blocking: spin up (or reuse) llama-server + warm the context on a
        background thread. Emits status/ready as it progresses."""
        threading.Thread(target=self._start_blocking, daemon=True).start()

    def _start_blocking(self):
        try:
            self.system_prompt = self.build_system_prompt()
            approx_tok = int(len(self.system_prompt) / 4)
            n = self.doc_count
            self._status(f"context pack ~{approx_tok} tokens "
                         f"({n} reference document{'' if n == 1 else 's'})")
            if approx_tok > self.cfg.ctx:
                self.emit("error", f"context pack (~{approx_tok} tokens) is larger than "
                                   f"companion_ctx ({self.cfg.ctx}); trim the documents "
                                   "or raise companion_ctx")

            if self._healthy():
                self._status(f"reusing llama-server already up on :{self.cfg.port}")
            elif not self._spawn_llama():
                return  # _spawn_llama emitted the error

            self._status("warming up model (priming context)...")
            if self._warmup():
                self.ready = True
                self.emit("ready", True)
                amb = "on" if self.ambient_enabled else "off"
                self._status(f"ready - ask a question or hit Talking points. Ambient notes: {amb}.")
                self._start_ambient_watcher()
        except Exception as e:
            self.emit("error", f"companion failed to start: {e}")

    def _spawn_llama(self) -> bool:
        if not self.cfg.llama_bin:
            self.emit("error", "llama-server not found on PATH - set companion_llama_bin "
                               "in settings.json")
            return False
        if not Path(self.cfg.llama_bin).is_file():
            self.emit("error", f"llama-server binary not found: {self.cfg.llama_bin}")
            return False
        if not self.cfg.model:
            self.emit("error", "no model configured - set companion_model in settings.json "
                               "to a GGUF chat model")
            return False
        if not Path(self.cfg.model).is_file():
            self.emit("error", f"model not found: {self.cfg.model}")
            return False

        cmd = [
            self.cfg.llama_bin,
            "-m", self.cfg.model,
            "--host", self.cfg.host,
            "--port", str(self.cfg.port),
            "-ngl", str(self.cfg.ngl),
            "-c", str(self.cfg.ctx),
            "-t", str(self.cfg.threads),
            "--jinja",
            "--no-webui",
        ]
        # The vendored build's RUNPATH points at its original absolute path; point
        # the loader at the binary's own dir where the ggml/llama .so files live.
        bin_dir = str(Path(self.cfg.llama_bin).resolve().parent)
        penv = {**os.environ}
        existing = penv.get("LD_LIBRARY_PATH", "")
        penv["LD_LIBRARY_PATH"] = bin_dir + (os.pathsep + existing if existing else "")

        self._status(f"starting llama-server on :{self.cfg.port}...")
        try:
            logf = open(self.cfg.log_path, "ab", buffering=0)
            self._proc = subprocess.Popen(cmd, stdout=logf, stderr=logf, env=penv)
        except OSError as e:
            self.emit("error", f"could not launch llama-server: {e}")
            return False
        self._spawned = True

        deadline = time.time() + self.cfg.ready_timeout
        while time.time() < deadline and not self._stopping:
            if self._proc.poll() is not None:
                self.emit("error",
                          f"llama-server exited early (code {self._proc.returncode}); "
                          f"see {self.cfg.log_path}")
                return False
            if self._healthy():
                self._status(f"llama-server ready on :{self.cfg.port}")
                return True
            time.sleep(0.6)
        if not self._stopping:
            self.emit("error", f"llama-server not ready within {self.cfg.ready_timeout:.0f}s; "
                               f"see {self.cfg.log_path}")
        return False

    def _warmup(self) -> bool:
        t0 = time.time()
        try:
            got = "".join(self._llm_stream(
                "Reply with exactly the word READY and nothing else.",
                temperature=0.0, max_tokens=8))
            self._status(f"context cached, model responsive ({time.time()-t0:.0f}s). "
                         f"Said: {got.strip()!r}")
            return True
        except Exception as e:
            self.emit("error", f"warmup failed ({e}); the model may still work - try a question.")
            return True  # non-fatal; let the user try anyway

    def stop(self):
        self._stopping = True
        self.ready = False
        if self._proc is not None and self._spawned and self._proc.poll() is None:
            self._status("stopping llama-server...")
            try:
                self._proc.terminate()
                self._proc.wait(timeout=8)
            except Exception:
                try:
                    self._proc.kill()
                except Exception:
                    pass
        self._proc = None

    # -- transcript ingest (append-only) ------------------------------
    def ingest(self, text: str):
        """Add a new transcript line (called in-process from the capture manager)."""
        if not text:
            return
        t = text.strip()
        if not t:
            return
        with self._lock:
            self._lines.append(t)
            self._last_speech_mono = time.monotonic()

    def reset(self, name: str = ""):
        with self._lock:
            self._lines = []
            self._responded_len = 0
            self._last_speech_mono = 0.0
            self._last_response_mono = 0.0

    def _transcript(self) -> str:
        with self._lock:
            text = "\n".join(self._lines)
        if len(text) > self.cfg.max_transcript_chars:
            text = "...\n" + text[-self.cfg.max_transcript_chars:]
        return text

    def _mark_responded(self):
        with self._lock:
            self._responded_len = len(self._lines)
            self._last_response_mono = time.monotonic()

    # -- LLM streaming ------------------------------------------------
    def _llm_stream(self, user_content: str, *, temperature=None, max_tokens=None):
        body = {
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_content},
            ],
            "temperature": self.cfg.temperature if temperature is None else temperature,
            "max_tokens": self.cfg.max_tokens if max_tokens is None else max_tokens,
            "stream": True,
        }
        data = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(
            f"{self.cfg.base_url}/v1/chat/completions",
            data=data, headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=300) as resp:
            for raw in resp:
                line = raw.decode("utf-8", errors="replace").strip()
                if not line or not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                try:
                    obj = json.loads(payload)
                    piece = obj["choices"][0].get("delta", {}).get("content")
                    if piece:
                        yield piece
                except (json.JSONDecodeError, KeyError, IndexError):
                    continue

    def _respond(self, header: str, instruction: str, *, max_tokens: int,
                 blocking: bool) -> bool:
        """Stream one answer built from the full running transcript + a trigger
        instruction. `blocking=False` (ambient) skips if a generation is in flight."""
        transcript = self._transcript()
        if not transcript.strip():
            self.emit("error", "no transcript yet - start recording first.")
            return False

        acquired = self._gen_lock.acquire(blocking=blocking)
        if not acquired:
            return False
        try:
            # Question-sandwich: state the task before the (potentially long) transcript
            # and restate it after, so the instruction stays in focus and the model
            # doesn't lose it in the middle of a large context.
            user = (
                f"[YOUR TASK] {instruction}\n\n"
                "[LIVE MEETING TRANSCRIPT so far, oldest to newest]\n"
                f"{transcript}\n\n"
                f"[YOUR TASK — restated] {instruction}\n"
                "Cite reference documents inline by source id. Follow the RULES."
            )
            self.emit("response_start", header)
            got = False
            try:
                for piece in self._llm_stream(user, max_tokens=max_tokens):
                    self.emit("token", piece)
                    got = True
            except (urllib.error.URLError, ConnectionError, TimeoutError) as e:
                self.emit("error", f"llm error: {e}")
                self.emit("response_end", None)
                return False
            if not got:
                self.emit("token", "(no output)")
            self.emit("response_end", None)
            self._mark_responded()
            return True
        finally:
            self._gen_lock.release()

    # -- public triggers (each runs on its own worker thread) ----------
    def ask_question(self, question: str):
        if not question.strip():
            return
        threading.Thread(
            target=self._respond,
            args=(f"ANSWER  ·  {question.strip()}", _question_instr(question)),
            kwargs={"max_tokens": 520, "blocking": True},
            daemon=True).start()

    def talking_points(self):
        threading.Thread(
            target=self._respond,
            args=("TALKING POINTS (current)", CURRENT_POINT_INSTR),
            kwargs={"max_tokens": 300, "blocking": True},
            daemon=True).start()

    def set_ambient(self, enabled: bool):
        self.ambient_enabled = enabled
        if enabled:
            self._mark_responded()  # don't dump a backlog note immediately

    # -- ambient pause watcher ----------------------------------------
    def _start_ambient_watcher(self):
        if self._ambient_thread and self._ambient_thread.is_alive():
            return
        self._ambient_thread = threading.Thread(target=self._ambient_loop, daemon=True)
        self._ambient_thread.start()

    def _ambient_loop(self):
        while not self._stopping:
            time.sleep(1.0)
            if not self.ambient_enabled or not self.ready:
                continue
            now = time.monotonic()
            with self._lock:
                last_speech = self._last_speech_mono
                last_resp = self._last_response_mono
                new_chars = len("\n".join(self._lines[self._responded_len:]))
            if last_speech == 0.0:
                continue
            if now - last_speech < self.cfg.pause_secs:       # still talking
                continue
            if now - last_resp < self.cfg.ambient_cooldown:   # too soon
                continue
            if new_chars < self.cfg.ambient_min_new_chars:    # not enough new speech
                continue
            self._respond("» ambient note", AMBIENT_INSTR, max_tokens=150, blocking=False)
