"""Common HTTP client for InfiniSynapse API calls.

Trimmed port of Spider2's ``spider_agent_infini.api.client``. Centralizes
credential loading, base URL handling, auth headers and request dispatch so
that :mod:`bird_agent_infini.api.database` only needs to describe the path
and payload.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import quote

import requests

DEFAULT_TIMEOUT: float = 10.0

# Default credential file: <repo>/infini/infini_credential.json
INFINI_CREDENTIAL_PATH = Path(__file__).resolve().parent.parent.parent / "infini_credential.json"

logger = logging.getLogger("bird_agent_infini")


def _load_credential(
    credential_path: str | os.PathLike | None = None,
) -> tuple[str, str, str | None]:
    """Load ``(api_url, api_key, console_url)`` from the credential JSON file.

    Environment variables override every field so the submission target can be
    switched without editing the JSON:

    - ``INFINI_CREDENTIAL_PATH``: path to the credential JSON to read (used only
      when no explicit ``credential_path`` argument is passed).
    - ``INFINI_API_URL``: overrides the runtime ``api_url``.
    - ``INFINI_CONSOLE_URL``: overrides the console ``console_url``.
    - ``INFINI_API_KEY``: overrides the ``api_key``.
    """
    if credential_path is not None:
        path = Path(credential_path)
    else:
        env_path = os.environ.get("INFINI_CREDENTIAL_PATH")
        path = Path(env_path) if env_path else INFINI_CREDENTIAL_PATH
    if path.is_file():
        with open(path, "r", encoding="utf-8") as f:
            cred = json.load(f)
    else:
        cred = {}

    api_url = os.environ.get("INFINI_API_URL") or cred.get("api_url")
    api_key = os.environ.get("INFINI_API_KEY") or cred.get("api_key")
    console_url = os.environ.get("INFINI_CONSOLE_URL") or cred.get("console_url")
    if not api_url or not api_key:
        raise ValueError(
            f"InfiniSynapse credential incomplete: need 'api_url' and 'api_key' in {path} "
            "(or INFINI_API_URL / INFINI_API_KEY environment variables). "
            "Copy infini_credential.json.example to infini_credential.json and fill it in."
        )
    return (
        api_url.rstrip("/"),
        api_key,
        console_url.rstrip("/") if isinstance(console_url, str) and console_url else None,
    )


class InfiniClient:
    """Thin wrapper around `requests` that injects base URL and Bearer auth."""

    def __init__(
        self,
        credential_path: str | os.PathLike | None = None,
        timeout: float = DEFAULT_TIMEOUT,
    ) -> None:
        self.api_url, self._api_key, self.console_url = _load_credential(credential_path)
        self.timeout = timeout

    def _headers(self, extra: Mapping[str, str] | None = None) -> dict[str, str]:
        headers = {
            "Accept": "application/json",
            "Authorization": f"Bearer {self._api_key}",
        }
        if extra:
            headers.update(extra)
        return headers

    def _url(self, path: str, *path_params: str) -> str:
        encoded = "/".join(quote(str(p), safe="") for p in path_params)
        path = path.rstrip("/")
        if encoded:
            path = f"{path}/{encoded}"
        if not path.startswith("/"):
            path = "/" + path
        return f"{self.api_url}{path}"

    def request(
        self,
        method: str,
        path: str,
        *path_params: str,
        params: Mapping[str, Any] | None = None,
        json_body: Any = None,
        headers: Mapping[str, str] | None = None,
        timeout: float | None = None,
        raise_for_status: bool = True,
    ) -> requests.Response:
        kwargs: dict[str, Any] = {
            "headers": self._headers(headers),
            "params": params,
            "timeout": timeout if timeout is not None else self.timeout,
            "json": json_body,
        }
        resp = requests.request(method, self._url(path, *path_params), **kwargs)
        if raise_for_status and resp.status_code >= 400 and resp.status_code != 404:
            resp.raise_for_status()
        return resp

    def get(self, path: str, *path_params: str, **kwargs: Any) -> requests.Response:
        return self.request("GET", path, *path_params, **kwargs)

    def post(self, path: str, *path_params: str, **kwargs: Any) -> requests.Response:
        return self.request("POST", path, *path_params, **kwargs)


def unwrap(body: Any) -> Any:
    """Unwrap nest-admin style `{code, data, message}` payloads."""
    if isinstance(body, dict) and "data" in body:
        return body["data"]
    return body
