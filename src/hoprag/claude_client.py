import json
import os
import subprocess
import time

from jsonschema import validate, ValidationError

_TEXT_FIELD = "result"  # confirmed in Task 4 Step 0

# The Claude Code CLI authenticates via the logged-in session. A stray/invalid
# ANTHROPIC_API_KEY (or AUTH_TOKEN) in the environment overrides that and causes
# 401s, so we drop these before invoking the CLI subprocess.
_DROP_ENV = ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN")


def _clean_env() -> dict:
    return {k: v for k, v in os.environ.items() if k not in _DROP_ENV}


def _strip_fence(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        t = t[3:]
        if t[:4].lower() == "json":
            t = t[4:]
        if t.endswith("```"):
            t = t[:-3]
    return t.strip()


class ClaudeError(Exception):
    pass


class ClaudeClient:
    def __init__(self, model: str | None = None, timeout: int = 120):
        self.model = model
        self.timeout = timeout

    def _run_raw(self, prompt: str) -> str:
        """Invoke `claude -p ... --output-format json`; return raw stdout. Retries on process error."""
        cmd = ["claude", "-p", prompt, "--output-format", "json"]
        if self.model:
            cmd += ["--model", self.model]
        last_err = None
        for attempt in range(3):
            try:
                proc = subprocess.run(
                    cmd, capture_output=True, text=True, timeout=self.timeout,
                    env=_clean_env(), stdin=subprocess.DEVNULL,
                    # the CLI emits UTF-8; force it (Windows would otherwise use the
                    # locale codec, e.g. cp950, and choke on em-dashes / curly quotes)
                    encoding="utf-8", errors="replace",
                )
                if proc.returncode == 0:
                    return proc.stdout
                last_err = proc.stderr
            except subprocess.TimeoutExpired as e:
                last_err = str(e)
            time.sleep(2 ** attempt)
        raise ClaudeError(f"claude CLI failed: {last_err}")

    def _extract_text(self, raw: str) -> str:
        obj = json.loads(raw)
        if obj.get("is_error"):
            raise ClaudeError(f"claude CLI returned an error: {obj.get(_TEXT_FIELD)}")
        return obj[_TEXT_FIELD]

    def complete(self, prompt: str) -> str:
        return self._extract_text(self._run_raw(prompt))

    def complete_json(self, prompt: str, schema: dict, max_retries: int = 3) -> dict:
        reminder = "\n\nReturn ONLY a JSON object, no prose, no code fences."
        last_err = None
        for _ in range(max_retries):
            text = self._extract_text(self._run_raw(prompt + reminder))
            try:
                data = json.loads(_strip_fence(text))
                validate(instance=data, schema=schema)
                return data
            except (json.JSONDecodeError, ValidationError) as e:
                last_err = e
        raise ClaudeError(f"invalid JSON after {max_retries} tries: {last_err}")
