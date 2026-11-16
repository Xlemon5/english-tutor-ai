"""HTTP-клиент веб-интерфейса: всё общение с ядром идёт через API."""
from __future__ import annotations

import json
import os
from collections.abc import Iterator

import httpx

API_URL = os.environ.get("API_URL", "http://localhost:8000").rstrip("/")
LONG = httpx.Timeout(600.0, connect=10.0)  # локальная модель может думать долго


class ApiError(Exception):
    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.message = message


class Api:
    def __init__(self, token: str = "") -> None:
        self.headers = {"Authorization": f"Bearer {token}"} if token else {}

    def _request(self, method: str, path: str, **kwargs) -> dict | list:
        try:
            r = httpx.request(method, API_URL + path, headers=self.headers, timeout=LONG, **kwargs)
        except httpx.HTTPError as e:
            raise ApiError(0, f"API недоступен ({API_URL}): {e}") from e
        if r.status_code >= 400:
            try:
                detail = r.json().get("detail", r.text)
            except ValueError:
                detail = r.text
            if isinstance(detail, list):  # ошибки валидации FastAPI
                detail = "; ".join(str(d.get("msg", d)) for d in detail)
            raise ApiError(r.status_code, str(detail))
        return r.json()

    def get(self, path: str) -> dict | list:
        return self._request("GET", path)

    def post(self, path: str, body: dict | None = None) -> dict:
        return self._request("POST", path, json=body or {})  # type: ignore[return-value]

    def delete(self, path: str) -> dict:
        return self._request("DELETE", path)  # type: ignore[return-value]

    def chat(self, message: str, level: str | None) -> Iterator[tuple[str, dict]]:
        """События SSE: (intent | token | final | error, данные)."""
        try:
            with httpx.stream("POST", API_URL + "/api/chat", headers=self.headers, timeout=LONG,
                              json={"message": message, "level": level}) as r:
                if r.status_code >= 400:
                    r.read()
                    raise ApiError(r.status_code, r.text)
                event = ""
                for line in r.iter_lines():
                    if line.startswith("event: "):
                        event = line[7:]
                    elif line.startswith("data: "):
                        yield event, json.loads(line[6:])
        except httpx.HTTPError as e:
            raise ApiError(0, f"API недоступен ({API_URL}): {e}") from e
