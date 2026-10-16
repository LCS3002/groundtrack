"""Desktop app: `groundtrack app`.

The whole workflow in one window: sites, footage and aerial, where the camera stood,
calibration and the area to track (both drawn in the app), runs with live logs, results,
combined plans, and the hand-over to Houdini (POP kits: build, open, add to the open scene)
and Rhino / Grasshopper (models, definitions, the groundtrack-grasshopper plugin).

It is the `groundtrack ui` server (same routes, same checks: this machine only, files inside
the project or a library folder you added) with more routes and a new page, shown in a native
window (pywebview: Edge WebView2 on Windows). Without pywebview it opens in the browser.
Every action still runs the normal `groundtrack` command, so the command line and the app
always do the same thing.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from . import desktop_cal as cal
from . import desktop_links as links
from .layout import Run, open_run
from .ui import NAME_RE, App, Job, _load, _run_root, make_handler

WEB = Path(__file__).with_name("desktop_web")
CTYPES = {".html": "text/html; charset=utf-8", ".css": "text/css; charset=utf-8",
          ".js": "application/javascript; charset=utf-8", ".woff2": "font/woff2",
          ".svg": "image/svg+xml", ".png": "image/png", ".txt": "text/plain; charset=utf-8"}
TEXT_EXT = {".py", ".txt", ".json", ".md", ".yaml", ".csv"}
NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


class NeedsConfirm(ValueError):
    """The action is possible but risky (e.g. writing a file Rhino may have open): ask first."""


def console_python() -> str:
    """python.exe next to pythonw.exe: jobs get a (hidden) console their own tools inherit,
    instead of each ffmpeg / hython popping up a window."""
    exe = Path(sys.executable)
    if exe.name.lower() == "pythonw.exe" and exe.with_name("python.exe").exists():
        return str(exe.with_name("python.exe"))
    return str(exe)


class AppJob(Job):
    """A Job run without a console window."""

    def __init__(self, kind: str, args: list[str], cwd: Path, label: str):  # noqa: D107
        self.id = uuid.uuid4().hex[:8]
        self.kind, self.args, self.label = kind, args, label
        self.lines: list[str] = []
        self.progress = ""
        self.started, self.ended = time.time(), None
        self.returncode: int | None = None
        self.cancelled = False
        env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")
        self.proc = subprocess.Popen([console_python(), "-m", "groundtrack", *args],
                                     cwd=str(cwd), stdout=subprocess.PIPE,
                                     stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, env=env,
                                     creationflags=NO_WINDOW)
        threading.Thread(target=self._read, daemon=True).start()


# --------------------------------------------------------------------------- state
class DesktopApp(App):
    job_class = AppJob

    def __init__(self, root: Path, settings: links.Settings | None = None):
        super().__init__(root)
        self.settings = settings or links.Settings()
        self.sites_cache = cal.Sites()
        self.device: dict = {"state": "checking"}
        self._roots: tuple[float, list[Path]] = (0.0, [])
        self._scan: tuple[float, dict | None] = (0.0, None)

    # ---- folders
    def libraries(self) -> list[Path]:
        """Where kits, models and definitions are looked for: the project's runs, the
        Grasshopper plugin's default kit folder, and the folders added in the app."""
        out = [self.root / "runs"]
        gh_default = Path.home() / "Documents" / "groundtrack houdini"
        if gh_default.is_dir():
            out.append(gh_default)
        out += [Path(p) for p in self.settings.libraries]
        seen, uniq = set(), []
        for p in out:
            if str(p) not in seen:
                seen.add(str(p))
                uniq.append(p)
        return uniq

    def allowed_roots(self) -> list[Path]:
        t, roots = self._roots
        if time.time() - t > 5:
            roots = super().allowed_roots() + [p.resolve() for p in self.libraries()]
            self._roots = (time.time(), roots)
        return roots

    def scan(self, refresh: bool = False) -> dict:
        t, data = self._scan
        if refresh or data is None or time.time() - t > 30:
            data = links.scan(self.libraries())
            self._scan = (time.time(), data)
        return data

    # ---- overview
    def overview(self) -> dict:
        return {"root": str(self.root), "sites": self.sites(), "device": self.device,
                "jobs": [j.summary() for j in self.jobs.values()],
                "running": links.running_apps()}

    def check_device(self) -> None:
        """`groundtrack device` once in the background (importing torch takes seconds)."""
        def work():
            try:
                r = subprocess.run([console_python(), "-m", "groundtrack", "device"],
                                   capture_output=True, text=True, timeout=180, cwd=str(self.root),
                                   creationflags=NO_WINDOW)
                d = {"state": "ok", "text": r.stdout.strip()}
                for ln in r.stdout.splitlines():
                    if ln.startswith("selected device:"):
                        d["device"] = ln.split(":", 1)[1].strip()
                    elif ln.startswith("GPU:"):
                        d["gpu"] = ln[4:].split(",")[0].strip()
                if "device" not in d:
                    d = {"state": "error", "text": (r.stdout + r.stderr).strip()[-400:]}
                self.device = d
            except (OSError, subprocess.SubprocessError) as e:
                self.device = {"state": "error", "text": str(e)}

        threading.Thread(target=work, daemon=True).start()

    def links(self, refresh: bool = False) -> dict:
        return {"houdini": links.houdini_install(), "rhino": links.rhino_install(),
                "grasshopper": links.grasshopper_plugin(), "running": links.running_apps(),
                "libraries": [{"path": str(p), "added": str(p) in self.settings.libraries}
                              for p in self.libraries()],
                **self.scan(refresh)}

    # ---- calibration / ROI
    def cal_site(self, rel: str) -> cal.Site:
        return self.sites_cache.get(self.resolve_config(rel))

    def cal_save(self, body: dict) -> dict:
        """Save the clicked pairs and fit the calibration with `groundtrack calibrate`."""
        rel = self.rel(self.resolve_config(body.get("config", "")))
        site = self.cal_site(rel)
        frame = int(body.get("frame", 0))
        csv = cal.save_pairs(site, body.get("pairs") or [])
        args = ["calibrate", "-c", rel, "--points-csv", str(csv), "--no-gui",
                "--frame", str(frame)]
        job = self.job_class("calibrate", args, self.root, f"calibrate · {site.cfg.site}")
        self.jobs[job.id] = job
        return job.summary()

    # ---- jobs
    def start_job(self, body: dict) -> dict:
        kind = body.get("kind")
        opts = body.get("options") or {}
        if kind == "rhino" and links.running_apps()["rhino"] and not opts.get("force"):
            raise NeedsConfirm("Rhino is open. If this run's model is open in it, close it "
                               "first: the model is rewritten.")
        return super().start_job(body)

    # ---- Houdini
    def clips(self) -> list[dict]:
        """Runs with a vector field (houdini/), newest first per site: kit modules."""
        out = []
        for f in self.config_files():
            try:
                cfg = _load(f)
            except Exception:  # noqa: BLE001 (a broken config is listed as such elsewhere)
                continue
            rd = _run_root(cfg)
            if not rd.exists():
                continue
            runs = sorted((p for p in rd.glob("*") if Run(p).is_run()),
                          key=lambda q: q.stat().st_mtime, reverse=True)
            for p in runs:
                r = open_run(p)
                if (r.houdini / "houdini_field.py").exists():
                    out.append({"config": self.rel(f), "site": cfg.site, "run": p.name,
                                "path": str(r.root)})
                    break
        return out

    def composites_all(self) -> list[dict]:
        comps = [{"name": c["name"], "path": str(self.root / "runs" / "composites" / c["name"])}
                 for c in self.composites()]
        have = {c["path"] for c in comps}
        for c in self.scan()["composites"]:
            if c["path"] not in have:
                p = Path(c["path"])
                comps.append({"name": f"{p.parent.name} / {p.name}", "path": c["path"]})
        return [c for c in comps if links.is_composite(Path(c["path"]))]

    def _kit_out(self, out: str) -> Path:
        p = Path(out).resolve()
        if not any(p == r or r in p.parents for r in self.allowed_roots()):
            raise PermissionError("the kit must go inside the project or a library folder")
        return p

    def _refuse_open_scene(self, kit: Path) -> None:
        has_scene = any(q.suffix.lower() in links.HIP_EXT for q in kit.glob("*")) \
            if kit.exists() else False
        if has_scene and links.running_apps()["houdini"]:
            raise ValueError("Houdini is open and this kit's scene may be open in it: close "
                             "Houdini first, or use 'Add to open scene' instead")

    def houdini_build(self, body: dict) -> dict:
        """A site's POP kit from its combined plan and clips (`groundtrack houdini-pop`)."""
        site = str(body.get("site", "")).strip()
        if not NAME_RE.match(site):
            raise ValueError("site: letters, numbers, - _ . only")
        comps = {c["path"] for c in self.composites_all()}
        comp = str(Path(body.get("composite", "")).resolve()) if body.get("composite") else ""
        if comp not in comps:
            raise ValueError("pick a combined plan")
        known = {c["path"]: c for c in self.clips()}
        clips = [str(Path(c).resolve()) for c in body.get("clips") or []]
        bad = [c for c in clips if c not in {str(Path(k).resolve()) for k in known}]
        if bad:
            raise ValueError(f"unknown clip {bad[0]}")
        out = self._kit_out(body.get("out") or str(self.root / "runs" / "houdini" / site))
        self._refuse_open_scene(out)
        args = ["houdini-pop", "--site", site, "--composite", comp, "--out", str(out)]
        if clips:
            args += ["--clips", *clips]
        if body.get("no_build"):
            args.append("--no-build")
        job = self.job_class("houdini-pop", args, self.root, f"houdini kit · {site}")
        self.jobs[job.id] = job
        return job.summary()

    def houdini_rebuild(self, body: dict) -> dict:
        """Build the kit's scene again in Houdini (hython), e.g. a kit from Grasshopper."""
        kit = self.check_file(body.get("kit", ""))
        if not links.is_kit(kit):
            raise ValueError("not a Houdini kit (modules.json + hou_pop_build.py)")
        self._refuse_open_scene(kit)
        job = self.job_class("houdini-build", ["houdini-build", str(kit)], self.root,
                             f"houdini scene · {kit.name}")
        self.jobs[job.id] = job
        return job.summary()

    # ---- files
    def open_file(self, body: dict) -> dict:
        links.open_file(self.check_file(body.get("path", "")))
        return {"ok": True}

    def text(self, path: str) -> dict:
        p = self.check_file(path)
        if p.suffix.lower() not in TEXT_EXT or p.stat().st_size > 2_000_000:
            raise ValueError("not a text file this app shows")
        return {"path": str(p), "text": p.read_text(encoding="utf-8", errors="replace")}

    def library(self, body: dict) -> dict:
        if body.get("action") == "remove":
            self.settings.remove_library(str(body.get("path", "")))
        else:
            self.settings.add_library(str(body.get("path", "")))
        self._roots = (0.0, [])
        return self.links(refresh=True)


# --------------------------------------------------------------------------- http
def make_desktop_handler(app: DesktopApp, port: int):
    Base = make_handler(app, port)

    post_routes = {
        "/api/job": app.start_job,
        "/api/cal/fit": None, "/api/cal/save": app.cal_save,
        "/api/roi/save": None, "/api/roi/clear": None,
        "/api/app/library": app.library, "/api/app/open-file": app.open_file,
        "/api/houdini/build": app.houdini_build, "/api/houdini/rebuild": app.houdini_rebuild,
    }

    class Handler(Base):
        server_version = "groundtrack-app"

        def _static(self, name: str):
            p = (WEB / name).resolve()
            if WEB.resolve() not in p.parents or not p.is_file():
                self._send(404, b"not found", "text/plain")
                return
            self._send(200, p.read_bytes(), CTYPES.get(p.suffix.lower(),
                                                        "application/octet-stream"))

        def do_GET(self):  # noqa: N802
            if not self._guard():
                return
            u = urlparse(self.path)
            q = {k: v[0] for k, v in parse_qs(u.query).items()}
            try:
                if u.path in ("/", "/index.html"):
                    self._static("index.html")
                elif u.path.startswith("/static/"):
                    self._static(u.path[len("/static/"):])
                elif u.path == "/api/app/overview":
                    self._json(app.overview())
                elif u.path == "/api/app/links":
                    self._json(app.links(q.get("refresh") == "1"))
                elif u.path == "/api/app/kit-sources":
                    self._json({"composites": app.composites_all(), "clips": app.clips()})
                elif u.path == "/api/app/text":
                    self._json(app.text(q["path"]))
                elif u.path == "/api/cal/state":
                    self._json(cal.state(app.cal_site(q["config"])))
                elif u.path == "/api/cal/frame":
                    data = cal.frame_jpeg(app.cal_site(q["config"]), int(q.get("frame", 0)),
                                          int(q.get("w", 0) or 0))
                    self._send(200, data, "image/jpeg")
                elif u.path == "/api/cal/map":
                    data = cal.map_jpeg(app.cal_site(q["config"]), int(q.get("w", 0) or 0))
                    self._send(200, data, "image/jpeg")
                else:
                    super().do_GET()
            except (KeyError, ValueError) as e:
                self._json({"error": str(e)}, 400)
            except PermissionError as e:
                self._json({"error": str(e)}, 403)
            except FileNotFoundError as e:
                self._json({"error": str(e)}, 404)
            except (ConnectionError, BrokenPipeError):
                pass
            except Exception as e:  # noqa: BLE001 (shown in the app, not a dropped connection)
                self._json({"error": f"{type(e).__name__}: {e}"}, 500)

        def do_POST(self):  # noqa: N802
            route = urlparse(self.path).path
            if route not in post_routes:
                super().do_POST()
                return
            if not self._guard():
                return
            if self.headers.get("X-Groundtrack") != "1":   # custom header: no cross-site forms
                self._send(403, b"forbidden", "text/plain")
                return
            n = int(self.headers.get("Content-Length") or 0)
            try:
                body = json.loads(self.rfile.read(n) or b"{}")
                if route == "/api/cal/fit":
                    site = app.cal_site(body.get("config", ""))
                    self._json(cal.fit(site, body.get("pairs") or [], body.get("query"),
                                       body.get("camera")))
                elif route == "/api/roi/save":
                    p = cal.save_roi(app.cal_site(body.get("config", "")), body.get("points"))
                    self._json({"ok": True, "path": str(p)})
                elif route == "/api/roi/clear":
                    b = cal.clear_roi(app.cal_site(body.get("config", "")))
                    self._json({"ok": True, "backup": str(b) if b else None})
                else:
                    self._json(post_routes[route](body))
            except NeedsConfirm as e:
                self._json({"error": str(e), "confirm": True}, 409)
            except PermissionError as e:
                self._json({"error": str(e)}, 403)
            except (KeyError, ValueError, OSError) as e:
                self._json({"error": str(e)}, 400)
            except Exception as e:  # noqa: BLE001
                self._json({"error": f"{type(e).__name__}: {e}"}, 500)

    return Handler


# --------------------------------------------------------------------------- window
class Bridge:
    """What the page may ask the window for (pywebview js_api): native file dialogs and,
    with the custom frame, the window buttons."""

    def __init__(self):
        self._window = None
        self._custom = False

    def frame(self) -> dict:
        from . import desktop_frame

        return {"custom": self._custom,
                "maximized": self._custom and desktop_frame.is_maximized(self._window)}

    def minimize(self) -> None:
        self._window.minimize()

    def toggle_maximize(self) -> None:
        from . import desktop_frame

        if desktop_frame.is_maximized(self._window):
            self._window.restore()
        else:
            self._window.maximize()

    def close(self) -> None:
        self._window.destroy()          # asks first while jobs run (the closing handler)

    def resize(self, edge: str) -> None:
        from . import desktop_frame

        if self._custom:
            desktop_frame.resize_from(self._window, str(edge))

    def pick(self, kind: str = "file", what: str = "") -> str | None:
        import webview

        types = {"video": ("Video (*.mp4;*.mov;*.MOV;*.m4v;*.avi)", "All files (*.*)"),
                 "geotiff": ("GeoTIFF (*.tif;*.tiff)", "All files (*.*)")}.get(what, ())
        dlg = getattr(webview, "FileDialog", None)
        if kind == "folder":
            mode = dlg.FOLDER if dlg else webview.FOLDER_DIALOG
            res = self._window.create_file_dialog(mode)
        else:
            mode = dlg.OPEN if dlg else webview.OPEN_DIALOG
            res = self._window.create_file_dialog(mode, file_types=types)
        if not res:
            return None
        return str(res[0] if isinstance(res, (list, tuple)) else res)


# the logo (desktop_web/index.html): three streamlines in ink on a neon tile, in its 32-unit
# box: (start, control 1, control 2, end), line width, opacity
LOGO_LINES = [(((4.5, 26), (12, 26), (15, 17), (27.5, 15)), 2.3, 0.35),
              (((4.5, 20.5), (11, 20.5), (14, 11.5), (27.5, 9.5)), 2.6, 1.0),
              (((4.5, 15), (10, 15), (12.5, 6.5), (21, 5)), 2.3, 0.6)]
ICON_FILE = "groundtrack-2.ico"          # a new name when the logo changes (it is cached)


def icon_path() -> Path | None:
    """The app icon (.ico), drawn once into the settings folder."""
    p = links.settings_dir() / ICON_FILE
    if p.exists():
        return p
    try:
        import numpy as np
        from PIL import Image, ImageDraw

        s = 32                                         # 1024 px, scaled down: smooth edges
        tile = Image.new("RGBA", (32 * s, 32 * s), (0, 0, 0, 0))
        ImageDraw.Draw(tile).rounded_rectangle((0, 0, 32 * s - 1, 32 * s - 1), radius=9 * s,
                                               fill=(57, 255, 136, 255))
        t = np.linspace(0, 1, 900)[:, None]
        for (p0, c1, c2, p1), width, alpha in LOGO_LINES:
            pts = ((1 - t) ** 3 * p0 + 3 * (1 - t) ** 2 * t * c1 + 3 * (1 - t) * t ** 2 * c2
                   + t ** 3 * p1) * s
            layer = Image.new("RGBA", tile.size, (0, 0, 0, 0))
            d = ImageDraw.Draw(layer)
            r = width * s / 2
            for x, y in pts:                          # stroke by stamping: no gaps, round caps
                d.ellipse((x - r, y - r, x + r, y + r), fill=(11, 11, 15, int(255 * alpha)))
            tile = Image.alpha_composite(tile, layer)
        im = tile.resize((256, 256), Image.Resampling.LANCZOS)
        p.parent.mkdir(parents=True, exist_ok=True)
        im.save(p, sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128),
                          (256, 256)])
        return p
    except Exception:  # noqa: BLE001 (no icon is fine)
        return None


def make_shortcut(root: Path) -> list[Path]:
    """Desktop and Start-menu shortcuts that open the app on this project (Windows)."""
    if not sys.platform.startswith("win"):
        raise SystemExit("shortcuts are made on Windows only")
    exe = Path(sys.executable)
    pyw = exe.with_name("pythonw.exe") if exe.with_name("pythonw.exe").exists() else exe
    ico = icon_path()
    targets = [Path.home() / "Desktop" / "groundtrack.lnk",
               Path(os.environ["APPDATA"]) / "Microsoft" / "Windows" / "Start Menu" / "Programs"
               / "groundtrack.lnk"]
    made = []
    for lnk in targets:
        if not lnk.parent.exists():
            continue
        ps = ("$s = (New-Object -ComObject WScript.Shell).CreateShortcut($env:GT_LNK); "
              "$s.TargetPath = $env:GT_EXE; $s.Arguments = $env:GT_ARGS; "
              "$s.WorkingDirectory = $env:GT_ROOT; $s.Description = 'groundtrack'; "
              "if ($env:GT_ICO) { $s.IconLocation = $env:GT_ICO }; $s.Save()")
        env = dict(os.environ, GT_LNK=str(lnk), GT_EXE=str(pyw), GT_ROOT=str(root),
                   GT_ARGS=f'-m groundtrack app --root "{root}"', GT_ICO=str(ico or ""))
        subprocess.run(["powershell", "-NoProfile", "-Command", ps], env=env, check=True,
                       creationflags=NO_WINDOW)
        made.append(lnk)
    return made


def _ask_quit(n: int) -> bool:
    """Native yes / no before closing while jobs run (they are stopped)."""
    if not sys.platform.startswith("win"):
        return True
    import ctypes

    jobs, them = ("jobs", "them") if n != 1 else ("job", "it")
    msg = (f"{n} {jobs} still running (a run, a calibration or a kit). Closing groundtrack "
           f"stops {them}.\n\nClose anyway?")
    return ctypes.windll.user32.MessageBoxW(None, msg, "groundtrack", 0x34) == 6  # yes / no


def serve(root: Path, port: int = 0, window: bool = True, debug: bool = False,
          custom_frame: bool = True) -> None:
    app = DesktopApp(root)
    srv = ThreadingHTTPServer(("127.0.0.1", port), BaseHTTPRequestHandler)
    port = srv.server_address[1]
    srv.RequestHandlerClass = make_desktop_handler(app, port)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{port}/"
    app.check_device()
    print(f"groundtrack app on {url}  (project {app.root})")
    try:
        try:
            import webview
        except ImportError:
            webview = None
        if not window or webview is None:
            if window:
                print("pywebview is not installed (pip install pywebview): opening the browser")
            import webbrowser

            if window:
                webbrowser.open(url)
            while True:
                time.sleep(3600)
        if sys.platform.startswith("win"):          # its own taskbar button and icon
            import ctypes

            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("groundtrack.app")
        bridge = Bridge()
        on_windows = sys.platform.startswith("win")
        # on Windows it appears once sized to the screen and framed (no title-bar flash)
        win = webview.create_window("groundtrack", url, js_api=bridge, width=1500, height=960,
                                    min_size=(1100, 700), background_color="#FFFFFF",
                                    text_select=True, hidden=on_windows)
        bridge._window = win
        shown = {"done": not on_windows}

        def show_once():
            if not shown["done"]:
                shown["done"] = True
                win.show()

        if on_windows:
            threading.Timer(15, show_once).start()       # whatever happens, show it

        def closing():
            n = sum(1 for j in app.jobs.values() if j.status == "running")
            return True if n == 0 else _ask_quit(n)

        win.events.closing += closing

        def push_frame():
            st = bridge.frame()
            win.evaluate_js(f"window.gtFrame && window.gtFrame({json.dumps(st)})")

        def loaded():
            from . import desktop_frame

            if not shown["done"]:
                desktop_frame.fit_to_screen(win, 1500, 960)
            if custom_frame and not bridge._custom:
                bridge._custom = desktop_frame.apply(win)
            push_frame()
            show_once()

        win.events.loaded += loaded
        win.events.maximized += push_frame
        win.events.restored += push_frame
        ico = icon_path()
        storage = links.settings_dir() / "webview"
        webview.start(gui="edgechromium" if sys.platform.startswith("win") else None,
                      debug=debug, private_mode=False, storage_path=str(storage),
                      icon=str(ico) if ico else None)
    except KeyboardInterrupt:
        pass
    finally:
        for j in app.jobs.values():
            j.stop()
        srv.shutdown()
        srv.server_close()
