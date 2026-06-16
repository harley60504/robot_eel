import json
from typing import Dict
from urllib.error import HTTPError, URLError
from urllib.request import urlopen


class OpenGoProHttpClient:
    def __init__(self, base_url: str = "http://10.5.5.9:8080", timeout: float = 8.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _url(self, path: str) -> str:
        if not path.startswith("/"):
            path = "/" + path
        return self.base_url + path

    def get_json(self, path: str) -> Dict:
        with urlopen(self._url(path), timeout=self.timeout) as response:
            data = response.read()
        return json.loads(data.decode("utf-8"))

    def command(self, path: str) -> Dict:
        try:
            return self.get_json(path)
        except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"GoPro command failed: {path}: {exc}") from exc

    def state(self) -> Dict:
        return self.command("/gopro/camera/state")

    def start_preview_stream(self, port: int = 8554) -> Dict:
        return self.command(f"/gopro/camera/stream/start?port={int(port)}")

    def stop_preview_stream(self) -> Dict:
        return self.command("/gopro/camera/stream/stop")
