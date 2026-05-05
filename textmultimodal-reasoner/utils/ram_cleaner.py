"""Windows RAM cleaner that mirrors RamMap's five clear actions.

Best-effort safety:
- skips current process and optional protected PIDs when emptying process working sets
- performs only OS cache/list cleanup APIs (no app-specific memory tampering)
"""

from __future__ import annotations

import argparse
import ctypes
import os
import platform
from ctypes import wintypes
from dataclasses import dataclass

SE_PROF_SINGLE_PROCESS_NAME = "SeProfileSingleProcessPrivilege"
SE_INCREASE_QUOTA_NAME = "SeIncreaseQuotaPrivilege"
TOKEN_ADJUST_PRIVILEGES = 0x0020
TOKEN_QUERY = 0x0008
SE_PRIVILEGE_ENABLED = 0x00000002

SYSTEM_FILE_CACHE_INFORMATION = 0x0015
SYSTEM_MEMORY_LIST_INFORMATION = 0x0050

MEMORY_EMPTY_WORKING_SETS = 0x2
MEMORY_FLUSH_MODIFIED_LIST = 0x3
MEMORY_PURGE_STANDBY_LIST = 0x4
MEMORY_PURGE_LOW_PRIORITY_STANDBY = 0x5

PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
PROCESS_SET_QUOTA = 0x0100


class SYSTEM_FILECACHE_INFORMATION(ctypes.Structure):
    _fields_ = [
        ("CurrentSize", ctypes.c_size_t),
        ("PeakSize", ctypes.c_size_t),
        ("PageFaultCount", wintypes.ULONG),
        ("MinimumWorkingSet", ctypes.c_size_t),
        ("MaximumWorkingSet", ctypes.c_size_t),
        ("CurrentSizeIncludingTransitionInPages", ctypes.c_size_t),
        ("PeakSizeIncludingTransitionInPages", ctypes.c_size_t),
        ("TransitionRePurposeCount", wintypes.ULONG),
        ("Flags", wintypes.ULONG),
    ]


@dataclass(slots=True)
class CleanResult:
    action: str
    success: bool
    details: str = ""


def _windows_only() -> None:
    if platform.system().lower() != "windows":
        raise RuntimeError("ram_cleaner.py only works on Windows.")


def enable_privilege(name: str) -> None:
    token = wintypes.HANDLE()
    if not ctypes.windll.advapi32.OpenProcessToken(
        ctypes.windll.kernel32.GetCurrentProcess(),
        TOKEN_ADJUST_PRIVILEGES | TOKEN_QUERY,
        ctypes.byref(token),
    ):
        raise OSError("OpenProcessToken failed")

    luid = wintypes.LUID()
    if not ctypes.windll.advapi32.LookupPrivilegeValueW(None, name, ctypes.byref(luid)):
        raise OSError(f"LookupPrivilegeValueW failed for {name}")

    class LUID_AND_ATTRIBUTES(ctypes.Structure):
        _fields_ = [("Luid", wintypes.LUID), ("Attributes", wintypes.DWORD)]

    class TOKEN_PRIVILEGES(ctypes.Structure):
        _fields_ = [("PrivilegeCount", wintypes.DWORD), ("Privileges", LUID_AND_ATTRIBUTES * 1)]

    tp = TOKEN_PRIVILEGES(1, (LUID_AND_ATTRIBUTES(luid, SE_PRIVILEGE_ENABLED),))
    if not ctypes.windll.advapi32.AdjustTokenPrivileges(token, False, ctypes.byref(tp), 0, None, None):
        raise OSError(f"AdjustTokenPrivileges failed for {name}")


def nt_set_system_information(info_class: int, payload: ctypes.Structure | ctypes.c_ulong) -> bool:
    status = ctypes.windll.ntdll.NtSetSystemInformation(
        info_class,
        ctypes.byref(payload),
        ctypes.sizeof(payload),
    )
    return status == 0


def empty_process_working_sets(protected_pids: set[int]) -> CleanResult:
    snap = ctypes.windll.kernel32.CreateToolhelp32Snapshot(0x00000002, 0)
    if snap == wintypes.HANDLE(-1).value:
        return CleanResult("Empty Working Sets", False, "snapshot failed")

    class PROCESSENTRY32W(ctypes.Structure):
        _fields_ = [
            ("dwSize", wintypes.DWORD),
            ("cntUsage", wintypes.DWORD),
            ("th32ProcessID", wintypes.DWORD),
            ("th32DefaultHeapID", ctypes.POINTER(ctypes.c_ulong)),
            ("th32ModuleID", wintypes.DWORD),
            ("cntThreads", wintypes.DWORD),
            ("th32ParentProcessID", wintypes.DWORD),
            ("pcPriClassBase", wintypes.LONG),
            ("dwFlags", wintypes.DWORD),
            ("szExeFile", wintypes.WCHAR * 260),
        ]

    entry = PROCESSENTRY32W()
    entry.dwSize = ctypes.sizeof(PROCESSENTRY32W)
    trimmed, skipped = 0, 0

    has_proc = ctypes.windll.kernel32.Process32FirstW(snap, ctypes.byref(entry))
    while has_proc:
        pid = int(entry.th32ProcessID)
        if pid in protected_pids or pid == 0:
            skipped += 1
        else:
            h_process = ctypes.windll.kernel32.OpenProcess(
                PROCESS_QUERY_LIMITED_INFORMATION | PROCESS_SET_QUOTA,
                False,
                pid,
            )
            if h_process:
                ctypes.windll.psapi.EmptyWorkingSet(h_process)
                ctypes.windll.kernel32.CloseHandle(h_process)
                trimmed += 1
            else:
                skipped += 1

        has_proc = ctypes.windll.kernel32.Process32NextW(snap, ctypes.byref(entry))

    ctypes.windll.kernel32.CloseHandle(snap)
    return CleanResult("Empty Working Sets", True, f"trimmed={trimmed}, skipped={skipped}")


def empty_system_working_set() -> CleanResult:
    info = SYSTEM_FILECACHE_INFORMATION()
    info.MinimumWorkingSet = ctypes.c_size_t(-1).value
    info.MaximumWorkingSet = ctypes.c_size_t(-1).value
    ok = nt_set_system_information(SYSTEM_FILE_CACHE_INFORMATION, info)
    return CleanResult("Empty System Working Set", ok)


def purge_list(command: int, label: str) -> CleanResult:
    cmd = ctypes.c_ulong(command)
    ok = nt_set_system_information(SYSTEM_MEMORY_LIST_INFORMATION, cmd)
    return CleanResult(label, ok)


def run_clean(protected_pids: set[int]) -> list[CleanResult]:
    _windows_only()
    enable_privilege(SE_PROF_SINGLE_PROCESS_NAME)
    enable_privilege(SE_INCREASE_QUOTA_NAME)

    return [
        empty_process_working_sets(protected_pids),
        empty_system_working_set(),
        purge_list(MEMORY_FLUSH_MODIFIED_LIST, "Empty Modified Page List"),
        purge_list(MEMORY_PURGE_STANDBY_LIST, "Empty Standby List"),
        purge_list(MEMORY_PURGE_LOW_PRIORITY_STANDBY, "Empty Priority 0 Standby List"),
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RamMap-like RAM cleaner for Windows")
    parser.add_argument(
        "--protect-pid",
        action="append",
        default=[],
        type=int,
        help="PID to exclude from process working-set trimming (repeatable).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    protected_pids = set(args.protect_pid)
    protected_pids.add(os.getpid())

    try:
        results = run_clean(protected_pids)
    except Exception as exc:
        print(f"RAM clean failed: {exc}")
        return 1

    for r in results:
        status = "Success" if r.success else "Failed"
        suffix = f" ({r.details})" if r.details else ""
        print(f"{r.action}: {status}{suffix}")

    return 0 if all(r.success for r in results) else 2


if __name__ == "__main__":
    raise SystemExit(main())
