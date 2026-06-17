import json
import subprocess
import time

from jsonschema import validate, ValidationError

_TEXT_FIELD = "result"  # confirmed in Task 4 Step 0


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
                    cmd, capture_output=True, text=True, timeout=self.timeout
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
        return obj[_TEXT_FIELD]

    def complete(self, prompt: str) -> str:
        return self._extract_text(self._run_raw(prompt))

    def complete_json(self, prompt: str, schema: dict, max_retries: int = 3) -> dict:
        reminder = "\n\nReturn ONLY a JSON object, no prose, no code fences."
        last_err = None
        for _ in range(max_retries):
            text = self._extract_text(self._run_raw(prompt + reminder))
            try:
                data = json.loads(text.strip().strip("`"))
                validate(instance=data, schema=schema)
                return data
            except (json.JSONDecodeError, ValidationError) as e:
                last_err = e
        raise ClaudeError(f"invalid JSON after {max_retries} tries: {last_err}")
