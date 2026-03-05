import json
import os
import threading
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

KEY_POOL = list("123456789abcdefghijklmnopqrstuvwxyz")


def _validate_project_name(project_name: str) -> str:
    value = (project_name or "").strip()
    if not value:
        raise ValueError("project 不能为空")
    if os.path.isabs(value):
        raise ValueError("project 不能是绝对路径")
    separators = [os.sep]
    if os.altsep:
        separators.append(os.altsep)
    if any(sep in value for sep in separators if sep):
        raise ValueError("project 仅支持项目名，不可包含路径分隔符")
    if value in {".", ".."}:
        raise ValueError("project 不能是 . 或 ..")
    return value


def _parse_script_payload(script_payload: Any) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if isinstance(script_payload, dict) and isinstance(script_payload.get("script"), list):
        characters = script_payload.get("characters", [])
        if characters is None:
            characters = []
        if not isinstance(characters, list):
            raise ValueError("script.json 中 characters 必须是数组")
        return characters, script_payload["script"]
    if isinstance(script_payload, list):
        return [], script_payload
    raise ValueError("script.json 格式不正确")


def _normalize_script_lines(script_lines: list[Any]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for line in script_lines:
        if isinstance(line, dict):
            normalized.append(line)
            continue
        normalized.append(
            {
                "text": "" if line is None else str(line),
                "speaker": "",
                "emotion": "neutral",
            }
        )
    return normalized


def _extract_character_list(raw: Any) -> list[dict[str, Any]]:
    if isinstance(raw, list):
        return [item for item in raw if isinstance(item, dict)]
    if isinstance(raw, dict) and isinstance(raw.get("characters"), list):
        return [item for item in raw["characters"] if isinstance(item, dict)]
    return []


@dataclass
class CachedProject:
    project_name: str
    project_root: Path
    script_path: Path
    payload: Any
    characters_from_script: list[dict[str, Any]]
    script_lines: list[dict[str, Any]]
    mtime_ns: int


class ScriptStore:
    def __init__(self, projects_root: Path):
        self._projects_root = projects_root
        self._lock = threading.RLock()
        self._cache: dict[str, CachedProject] = {}

    def list_projects(self) -> list[str]:
        if not self._projects_root.exists():
            return []
        projects: list[str] = []
        for entry in self._projects_root.iterdir():
            if not entry.is_dir():
                continue
            if (entry / "script.json").is_file():
                projects.append(entry.name)
        projects.sort()
        return projects

    def get_project_view(self, project_name: str) -> dict[str, Any]:
        with self._lock:
            cached = self._load_project_unlocked(project_name)
            characters = self._load_characters_unlocked(cached)
            key_map = self._build_key_map(characters)
            return {
                "project": cached.project_name,
                "script_path": str(cached.script_path),
                "total_lines": len(cached.script_lines),
                "characters": characters,
                "key_map": key_map,
                "lines": cached.script_lines,
            }

    def update_line(self, project_name: str, line_index: int, patch: dict[str, Any]) -> dict[str, Any]:
        allowed_fields = {"text", "speaker", "emotion", "type"}
        filtered_patch: dict[str, str] = {}
        for key, value in patch.items():
            if key in allowed_fields:
                filtered_patch[key] = "" if value is None else str(value)
        if not filtered_patch:
            raise ValueError("至少需要提供 text/speaker/emotion/type 之一")

        with self._lock:
            cached = self._load_project_unlocked(project_name)
            if line_index < 0 or line_index >= len(cached.script_lines):
                raise IndexError("line_index 超出范围")
            target = cached.script_lines[line_index]
            target.update(filtered_patch)
            self._write_project_unlocked(cached)
            return {
                "ok": True,
                "line_index": line_index,
                "line": target,
            }

    def insert_line_after(self, project_name: str, line_index: int) -> dict[str, Any]:
        with self._lock:
            cached = self._load_project_unlocked(project_name)
            total = len(cached.script_lines)
            if total == 0 and line_index == -1:
                insert_at = 0
            elif line_index < 0 or line_index >= total:
                raise IndexError("line_index 超出范围")
            else:
                insert_at = line_index + 1

            new_line: dict[str, str] = {
                "text": "",
                "speaker": "",
                "emotion": "",
                "type": "",
            }
            cached.script_lines.insert(insert_at, new_line)
            self._write_project_unlocked(cached)
            return {
                "ok": True,
                "inserted_index": insert_at,
                "line": new_line,
                "total_lines": len(cached.script_lines),
            }

    def _load_project_unlocked(self, project_name: str) -> CachedProject:
        valid_name = _validate_project_name(project_name)
        project_root = self._projects_root / valid_name
        script_path = project_root / "script.json"
        if not script_path.is_file():
            raise FileNotFoundError(f"未找到剧本文件: {script_path}")

        mtime_ns = script_path.stat().st_mtime_ns
        cached = self._cache.get(valid_name)
        if cached and cached.mtime_ns == mtime_ns:
            return cached

        with script_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        characters_from_script, script_lines_raw = _parse_script_payload(payload)
        script_lines = _normalize_script_lines(script_lines_raw)
        if isinstance(payload, dict):
            payload["script"] = script_lines
        else:
            payload = script_lines

        cached_project = CachedProject(
            project_name=valid_name,
            project_root=project_root,
            script_path=script_path,
            payload=payload,
            characters_from_script=characters_from_script,
            script_lines=script_lines,
            mtime_ns=mtime_ns,
        )
        self._cache[valid_name] = cached_project
        return cached_project

    def _load_characters_unlocked(self, project: CachedProject) -> list[dict[str, Any]]:
        character_candidates: list[dict[str, Any]] = []
        for file_name in ("characters.json", "character_db.json"):
            candidate_path = project.project_root / file_name
            if not candidate_path.is_file():
                continue
            try:
                with candidate_path.open("r", encoding="utf-8") as f:
                    raw = json.load(f)
                character_candidates.extend(_extract_character_list(raw))
            except json.JSONDecodeError:
                continue

        if not character_candidates:
            character_candidates.extend(project.characters_from_script)

        deduped: list[dict[str, Any]] = []
        seen_names: set[str] = set()
        for item in character_candidates:
            name = str(item.get("name", "")).strip()
            if not name or name in seen_names:
                continue
            seen_names.add(name)
            deduped.append(item)
        return deduped

    def _write_project_unlocked(self, project: CachedProject) -> None:
        tmp_path = project.script_path.with_suffix(".json.tmp")
        with tmp_path.open("w", encoding="utf-8") as f:
            json.dump(project.payload, f, ensure_ascii=False, indent=2)
            f.write("\n")
        os.replace(tmp_path, project.script_path)
        project.mtime_ns = project.script_path.stat().st_mtime_ns

    @staticmethod
    def _build_key_map(characters: list[dict[str, Any]]) -> list[dict[str, str]]:
        key_map: list[dict[str, str]] = [{"key": "0", "name": "narrator"}]
        seen_names = {"narrator"}
        for idx, character in enumerate(characters):
            if idx >= len(KEY_POOL):
                break
            name = str(character.get("name", "")).strip()
            if not name:
                continue
            if name.lower() in seen_names:
                continue
            seen_names.add(name.lower())
            key_map.append({"key": KEY_POOL[idx], "name": name})
        return key_map


def _json_response(handler: BaseHTTPRequestHandler, status: int, payload: Any) -> None:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def _text_response(handler: BaseHTTPRequestHandler, status: int, text: str) -> None:
    body = text.encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "text/plain; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def _static_content_type(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".html":
        return "text/html; charset=utf-8"
    if suffix == ".js":
        return "application/javascript; charset=utf-8"
    if suffix == ".css":
        return "text/css; charset=utf-8"
    return "application/octet-stream"


def run_proofread_server(
    project_name: str,
    *,
    host: str = "127.0.0.1",
    port: int = 8765,
    projects_root: str = "data/projects",
) -> None:
    _validate_project_name(project_name)
    root_path = Path(projects_root).resolve()
    store = ScriptStore(root_path)
    _ = store.get_project_view(project_name)

    static_root = Path(__file__).resolve().parent / "web" / "proofread"
    if not static_root.is_dir():
        raise FileNotFoundError(f"静态资源目录不存在: {static_root}")

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            path = unquote(parsed.path)
            if path == "/api/health":
                _json_response(self, HTTPStatus.OK, {"ok": True})
                return
            if path == "/api/projects":
                _json_response(
                    self,
                    HTTPStatus.OK,
                    {"projects": store.list_projects(), "default_project": project_name},
                )
                return
            if path.startswith("/api/project/"):
                self._handle_get_project(path)
                return
            self._handle_static(path)

        def do_PATCH(self) -> None:
            parsed = urlparse(self.path)
            path = unquote(parsed.path)
            if not path.startswith("/api/project/"):
                _json_response(self, HTTPStatus.NOT_FOUND, {"error": "Not Found"})
                return

            segments = path.split("/")
            if len(segments) != 6 or segments[4] != "line":
                _json_response(self, HTTPStatus.NOT_FOUND, {"error": "Not Found"})
                return

            target_project = segments[3]
            try:
                line_index = int(segments[5])
            except ValueError:
                _json_response(self, HTTPStatus.BAD_REQUEST, {"error": "line index 非法"})
                return

            try:
                body = self._read_json_body()
                result = store.update_line(target_project, line_index, body)
                _json_response(self, HTTPStatus.OK, result)
            except FileNotFoundError as exc:
                _json_response(self, HTTPStatus.NOT_FOUND, {"error": str(exc)})
            except (ValueError, IndexError) as exc:
                _json_response(self, HTTPStatus.BAD_REQUEST, {"error": str(exc)})
            except json.JSONDecodeError:
                _json_response(self, HTTPStatus.BAD_REQUEST, {"error": "请求体必须是 JSON"})
            except Exception as exc:  # noqa: BLE001
                _json_response(self, HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(exc)})

        def do_POST(self) -> None:
            parsed = urlparse(self.path)
            path = unquote(parsed.path)
            if not path.startswith("/api/project/"):
                _json_response(self, HTTPStatus.NOT_FOUND, {"error": "Not Found"})
                return

            segments = path.split("/")
            if len(segments) != 7 or segments[4] != "line" or segments[6] != "insert":
                _json_response(self, HTTPStatus.NOT_FOUND, {"error": "Not Found"})
                return

            target_project = segments[3]
            try:
                line_index = int(segments[5])
            except ValueError:
                _json_response(self, HTTPStatus.BAD_REQUEST, {"error": "line index 非法"})
                return

            try:
                result = store.insert_line_after(target_project, line_index)
                _json_response(self, HTTPStatus.OK, result)
            except FileNotFoundError as exc:
                _json_response(self, HTTPStatus.NOT_FOUND, {"error": str(exc)})
            except (ValueError, IndexError) as exc:
                _json_response(self, HTTPStatus.BAD_REQUEST, {"error": str(exc)})
            except Exception as exc:  # noqa: BLE001
                _json_response(self, HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(exc)})

        def _handle_get_project(self, path: str) -> None:
            segments = path.split("/")
            if len(segments) != 4:
                _json_response(self, HTTPStatus.NOT_FOUND, {"error": "Not Found"})
                return
            target_project = segments[3]
            try:
                payload = store.get_project_view(target_project)
                _json_response(self, HTTPStatus.OK, payload)
            except FileNotFoundError as exc:
                _json_response(self, HTTPStatus.NOT_FOUND, {"error": str(exc)})
            except ValueError as exc:
                _json_response(self, HTTPStatus.BAD_REQUEST, {"error": str(exc)})
            except Exception as exc:  # noqa: BLE001
                _json_response(self, HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(exc)})

        def _handle_static(self, path: str) -> None:
            if path in {"", "/"}:
                rel = "index.html"
            else:
                rel = path.lstrip("/")
            file_path = (static_root / rel).resolve()
            if static_root not in file_path.parents and file_path != static_root:
                _text_response(self, HTTPStatus.FORBIDDEN, "Forbidden")
                return
            if not file_path.is_file():
                _text_response(self, HTTPStatus.NOT_FOUND, "Not Found")
                return
            data = file_path.read_bytes()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", _static_content_type(file_path))
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _read_json_body(self) -> dict[str, Any]:
            content_length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(content_length)
            data = json.loads(raw.decode("utf-8") if raw else "{}")
            if not isinstance(data, dict):
                raise ValueError("请求体必须是 JSON 对象")
            return data

        def log_message(self, fmt: str, *args: Any) -> None:
            message = fmt % args
            print(f"[proofread-web] {message}")

    server = ThreadingHTTPServer((host, port), Handler)
    url = f"http://{host}:{port}/?project={project_name}"
    print(f"🧪 剧本校对界面已启动: {url}")
    print("按 Ctrl+C 停止服务。")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
