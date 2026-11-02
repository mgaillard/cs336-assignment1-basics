#!/usr/bin/env python3
"""Audit a 32-bit Windows .exe for Windows 98 SE compatibility (no dependencies, no Windows needed).

Checks that the PE file
  * is an i386 console executable that asks for OS / subsystem version <= 4.0 (Windows 95/98/NT4);
  * imports only from KERNEL32.dll and msvcrt.dll (both ship with Windows 98 SE);
  * imports only functions from the reviewed allow-lists below.

The allow-lists are the functions the MinGW-w64 C runtime plus src/platform_win32.c use today; each of
them is exported by Windows 98 SE. A new import fails the check until it has been reviewed (look it up in
the Win32 documentation: "Minimum supported client" must be Windows 95/98, not 2000/XP or later) and added.

    python tools/check_imports.py build/x86_sse_win/cs336_infer.exe
"""

import struct
import sys

ALLOWED = {
    "kernel32.dll": {
        # src/platform_win32.c
        "CloseHandle", "CreateFileA", "ReadFile", "SetFilePointer", "GetLastError", "GetTickCount",
        "GetCurrentProcessId", "QueryPerformanceCounter", "QueryPerformanceFrequency",
        # MinGW-w64 CRT start-up, exception filter, TLS and pseudo-relocation support
        "DeleteCriticalSection", "EnterCriticalSection", "InitializeCriticalSection", "LeaveCriticalSection",
        "FreeLibrary", "GetModuleHandleA", "GetProcAddress", "LoadLibraryA", "Sleep", "TlsGetValue",
        "SetUnhandledExceptionFilter", "VirtualProtect", "VirtualQuery", "IsDBCSLeadByteEx",
        "MultiByteToWideChar", "WideCharToMultiByte",
    },
    "msvcrt.dll": {
        "__getmainargs", "__initenv", "__mb_cur_max", "__p__commode", "__p__fmode", "__set_app_type",
        "__setusermatherr", "_amsg_exit", "_cexit", "_errno", "_initterm", "_iob", "_lock", "_onexit", "_unlock",
        "abort", "atof", "atoi", "calloc", "exit", "fclose", "ferror", "fflush", "fopen", "fprintf", "fputc",
        "fputs", "fread", "free", "fwrite", "localeconv", "malloc", "memcmp", "memcpy", "memmove", "memset",
        "qsort", "realloc", "setlocale", "signal", "strchr", "strcmp", "strerror", "strlen", "strncmp",
        "strtoul", "vfprintf", "wcslen",
    },
}

# Present in KERNEL32.dll of Windows 9x only as a stub returning ERROR_CALL_NOT_IMPLEMENTED; the MinGW CRT
# calls it only to look up its own module, so it is tolerated but reported.
WIN9X_STUBS = {"GetModuleHandleW"}


def parse_pe(path):
    with open(path, "rb") as f:
        data = f.read()
    if data[:2] != b"MZ":
        raise ValueError("not a PE file")
    pe = struct.unpack_from("<I", data, 0x3C)[0]
    if data[pe : pe + 4] != b"PE\0\0":
        raise ValueError("missing PE signature")
    machine, nsections, _, _, _, opt_size, _ = struct.unpack_from("<HHIIIHH", data, pe + 4)
    opt = pe + 24
    magic = struct.unpack_from("<H", data, opt)[0]
    if magic != 0x10B:
        raise ValueError("not a PE32 (32-bit) executable")
    os_major, os_minor = struct.unpack_from("<HH", data, opt + 40)
    sub_major, sub_minor = struct.unpack_from("<HH", data, opt + 48)
    subsystem = struct.unpack_from("<H", data, opt + 68)[0]
    import_rva = struct.unpack_from("<I", data, opt + 96 + 8)[0]
    sections = []
    sec = opt + opt_size
    for i in range(nsections):
        vsize, vaddr, rsize, raw = struct.unpack_from("<IIII", data, sec + 40 * i + 8)
        sections.append((vaddr, max(vsize, rsize), raw))

    def offset(rva):
        for vaddr, size, raw in sections:
            if vaddr <= rva < vaddr + size:
                return raw + rva - vaddr
        raise ValueError(f"RVA {rva:#x} is outside every section")

    def cstr(rva):
        start = offset(rva)
        return data[start : data.index(b"\0", start)].decode("ascii")

    imports = {}
    if import_rva:
        desc = offset(import_rva)
        while True:
            oft, _, _, name_rva, ft = struct.unpack_from("<IIIII", data, desc)
            if name_rva == 0:
                break
            names = imports.setdefault(cstr(name_rva).lower(), set())
            thunk = offset(oft or ft)
            while True:
                value = struct.unpack_from("<I", data, thunk)[0]
                if value == 0:
                    break
                names.add(f"#{value & 0xFFFF}" if value & 0x80000000 else cstr((value & 0x7FFFFFFF) + 2))
                thunk += 4
            desc += 20
    return {"machine": machine, "os": (os_major, os_minor), "subsystem_version": (sub_major, sub_minor),
            "subsystem": subsystem, "imports": imports}


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    info = parse_pe(sys.argv[1])
    problems = []
    if info["machine"] != 0x14C:
        problems.append(f"machine type {info['machine']:#x} is not i386")
    if info["subsystem"] != 3:
        problems.append("not a console executable")
    for label, version in (("OS", info["os"]), ("subsystem", info["subsystem_version"])):
        if version > (4, 0):
            problems.append(f"required {label} version {version[0]}.{version[1]} is newer than 4.0 (Windows 95/98/NT4)")
    for dll, names in sorted(info["imports"].items()):
        allowed = ALLOWED.get(dll)
        print(f"{dll}: {len(names)} imports")
        if allowed is None:
            problems.append(f"{dll} is not a DLL of the Windows 98 SE allow-list (kernel32.dll, msvcrt.dll)")
            continue
        for name in sorted(names):
            if name in WIN9X_STUBS:
                print(f"  warning: {dll}!{name} is only a stub on Windows 9x")
            elif name not in allowed:
                problems.append(f"{dll}!{name} has not been reviewed as available on Windows 98 SE")
    if problems:
        print("FAIL:")
        for p in problems:
            print("  -", p)
        sys.exit(1)
    print(f"OK: i386 console exe, OS/subsystem version {info['os'][0]}.{info['os'][1]}, imports match the allow-list")


if __name__ == "__main__":
    main()
