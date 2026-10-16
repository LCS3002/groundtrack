"""A custom window frame for the desktop app on Windows (like Cursor or VS Code): the app's
own top bar is the title bar, with its own minimise / maximise / close buttons.

It stays a normal Windows window: only the caption is removed (WM_NCCALCSIZE), so the shadow,
the resize borders, Snap and Win+arrow keep working. WebView2's non-client region support makes
the page's `app-region: drag` areas behave as the title bar (drag, double-click to maximise,
drag to the top to snap). The top edge, which no longer has a border, resizes through the same
message Windows sends for a border (WM_NCLBUTTONDOWN + HTTOP), started from the page.

If anything here fails the window simply keeps its normal title bar.
"""

from __future__ import annotations

import ctypes
import sys
from ctypes import wintypes

WM_NCCALCSIZE = 0x0083
WM_NCLBUTTONDOWN = 0x00A1
SM_CYFRAME, SM_CXPADDEDBORDER = 33, 92
SWP_FRAMECHANGED = 0x0020
SWP_FLAGS = 0x0001 | 0x0002 | 0x0004 | 0x0010 | SWP_FRAMECHANGED   # no size/move/z/activate
HIT = {"top": 12, "topleft": 13, "topright": 14}
SUBCLASS_ID = 0x67746B                                                # "gtk"

LRESULT = ctypes.c_ssize_t
SUBCLASSPROC = ctypes.WINFUNCTYPE(LRESULT, wintypes.HWND, wintypes.UINT, ctypes.c_size_t,
                                  ctypes.c_ssize_t, ctypes.c_size_t, ctypes.c_size_t)


class NCCALCSIZE_PARAMS(ctypes.Structure):  # noqa: N801 (the Windows name)
    _fields_ = [("rgrc", wintypes.RECT * 3), ("lppos", ctypes.c_void_p)]


_keep: dict = {}          # the callback must outlive this module's functions


def _dlls():
    user32, comctl32 = ctypes.windll.user32, ctypes.windll.comctl32
    comctl32.DefSubclassProc.argtypes = [wintypes.HWND, wintypes.UINT, ctypes.c_size_t,
                                         ctypes.c_ssize_t]
    comctl32.DefSubclassProc.restype = LRESULT
    comctl32.SetWindowSubclass.argtypes = [wintypes.HWND, SUBCLASSPROC, ctypes.c_size_t,
                                           ctypes.c_size_t]
    comctl32.SetWindowSubclass.restype = wintypes.BOOL
    user32.SendMessageW.argtypes = [wintypes.HWND, wintypes.UINT, ctypes.c_size_t,
                                    ctypes.c_ssize_t]
    user32.SendMessageW.restype = LRESULT
    user32.SetWindowPos.argtypes = [wintypes.HWND, wintypes.HWND, ctypes.c_int, ctypes.c_int,
                                    ctypes.c_int, ctypes.c_int, wintypes.UINT]
    user32.IsZoomed.argtypes = [wintypes.HWND]
    return user32, comctl32


def _frame_px(user32, hwnd) -> int:
    """How far a maximised window overhangs the screen at the top (its resize border)."""
    try:
        dpi = user32.GetDpiForWindow(hwnd)
        return (user32.GetSystemMetricsForDpi(SM_CYFRAME, dpi)
                + user32.GetSystemMetricsForDpi(SM_CXPADDEDBORDER, dpi))
    except AttributeError:
        return user32.GetSystemMetrics(SM_CYFRAME) + user32.GetSystemMetrics(SM_CXPADDEDBORDER)


def _invoke(form, fn, wait: bool = True):
    from System import Action

    if wait:
        form.Invoke(Action(fn))
    else:
        form.BeginInvoke(Action(fn))


def hwnd_of(window) -> int:
    return int(window.native.Handle.ToInt64())


def apply(window) -> bool:
    """Remove the caption of a shown pywebview window (WinForms + Edge). True when it worked."""
    if not sys.platform.startswith("win") or getattr(window, "native", None) is None:
        return False
    form = window.native
    done = {"ok": False}

    def on_ui():
        browser = getattr(form, "browser", None)
        core = getattr(getattr(browser, "webview", None), "CoreWebView2", None)
        if core is None:
            return
        core.Settings.IsNonClientRegionSupportEnabled = True    # app-region: drag = title bar
        user32, comctl32 = _dlls()
        hwnd = int(form.Handle.ToInt64())

        @SUBCLASSPROC
        def proc(h, msg, wp, lp, _uid, _ref):
            if msg == WM_NCCALCSIZE and wp:
                p = ctypes.cast(lp, ctypes.POINTER(NCCALCSIZE_PARAMS)).contents
                top = p.rgrc[0].top
                res = comctl32.DefSubclassProc(h, msg, wp, lp)     # sides and bottom as usual
                p.rgrc[0].top = top + (_frame_px(user32, h) if user32.IsZoomed(h) else 0)
                return res
            return comctl32.DefSubclassProc(h, msg, wp, lp)

        if not comctl32.SetWindowSubclass(hwnd, proc, SUBCLASS_ID, 0):
            return
        _keep[hwnd] = proc
        user32.SetWindowPos(hwnd, None, 0, 0, 0, 0, SWP_FLAGS)     # recalculate the frame now
        done["ok"] = True

    try:
        _invoke(form, on_ui)
    except Exception:  # noqa: BLE001 (keep the normal title bar)
        return False
    return done["ok"]


def fit_to_screen(window, width: int, height: int, share: float = 0.94) -> None:
    """Size the window to width x height (logical px), at most `share` of the work area of
    its monitor (above the taskbar), centred."""
    if not sys.platform.startswith("win") or getattr(window, "native", None) is None:
        return
    form = window.native

    def on_ui():
        from System.Drawing import Rectangle
        from System.Windows.Forms import Screen

        user32, _ = _dlls()
        hwnd = int(form.Handle.ToInt64())
        try:
            scale = user32.GetDpiForWindow(hwnd) / 96.0
        except AttributeError:
            scale = 1.0
        wa = Screen.FromHandle(form.Handle).WorkingArea
        w = min(int(width * scale), int(wa.Width * share))
        h = min(int(height * scale), int(wa.Height * share))
        form.Bounds = Rectangle(wa.X + (wa.Width - w) // 2, wa.Y + (wa.Height - h) // 2, w, h)

    try:
        _invoke(form, on_ui)
    except Exception:  # noqa: BLE001 (the default size then)
        pass


def is_maximized(window) -> bool:
    user32, _ = _dlls()
    return bool(user32.IsZoomed(hwnd_of(window)))


def resize_from(window, edge: str) -> None:
    """Start a native resize from the top edge / corners (the mouse button is still down)."""
    if edge not in HIT:
        return
    user32, _ = _dlls()
    hwnd = hwnd_of(window)

    def on_ui():
        user32.ReleaseCapture()
        user32.SendMessageW(hwnd, WM_NCLBUTTONDOWN, HIT[edge], 0)

    _invoke(window.native, on_ui, wait=False)
