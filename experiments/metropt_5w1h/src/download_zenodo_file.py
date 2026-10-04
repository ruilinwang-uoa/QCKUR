"""Generic resume-capable Zenodo file downloader (record, filename, dest).

Zenodo supports Range (unlike UCI, observed 2026-09-20) and drops
mid-transfer on this network, so every fetch goes through a Range-resume
loop. Also usable head-only via --probe to print Content-Range/length.

Usage:
    python code/src/download_zenodo_file.py RECORD FILENAME DEST
    python code/src/download_zenodo_file.py RECORD FILENAME DEST --probe
"""

from __future__ import annotations

import argparse
import sys
import time
import urllib.request
from pathlib import Path


def url_for(record: int, name: str) -> str:
    return f"https://zenodo.org/api/records/{record}/files/{name}/content"


def download(url: str, dest: Path) -> None:
    tmp = dest.with_suffix(dest.suffix + ".part")
    attempts, last_gb = 0, -1
    while attempts < 60:
        attempts += 1
        have = tmp.stat().st_size if tmp.exists() else 0
        headers = {"Range": f"bytes={have}-"} if have else {}
        req = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                if have and getattr(r, "status", 200) != 206:
                    print("server ignored Range -- restarting from 0")
                    tmp.unlink(missing_ok=True)
                    continue
                total = r.headers.get("Content-Range", "").split("/")[-1]
                with open(tmp, "ab" if have else "wb") as f:
                    while True:
                        chunk = r.read(1 << 20)
                        if not chunk:
                            break
                        f.write(chunk)
                        gb = tmp.stat().st_size >> 30
                        if gb != last_gb:
                            last_gb = gb
                            done = tmp.stat().st_size
                            print(f"  {done >> 20} MB"
                                  + (f" / {int(total) >> 20} MB"
                                     if total.isdigit() else ""), flush=True)
        except urllib.error.HTTPError as e:
            if e.code == 416:  # requested range past EOF = already complete
                have = tmp.stat().st_size if tmp.exists() else 0
                rq = urllib.request.Request(url, method="HEAD")
                with urllib.request.urlopen(rq, timeout=60) as r:
                    total = int(r.headers.get("Content-Length", 0))
                if total and have >= total:
                    break
            print(f"  HTTP {e.code}; resume in 10 s ({attempts}/60)",
                  flush=True)
            time.sleep(10)
        except Exception as e:  # noqa: BLE001
            got = tmp.stat().st_size if tmp.exists() else 0
            print(f"  error at {got >> 20} MB ({e}); resume in 10 s "
                  f"({attempts}/60)", flush=True)
            time.sleep(10)
    else:
        sys.exit("download failed after 60 attempts")
    tmp.rename(dest)
    print(f"complete: {dest} ({dest.stat().st_size >> 20} MB)", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("record", type=int)
    ap.add_argument("filename")
    ap.add_argument("dest", type=Path)
    ap.add_argument("--probe", action="store_true",
                    help="HEAD only: print size, no download")
    args = ap.parse_args()

    url = url_for(args.record, args.filename)
    if args.probe:
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req, timeout=60) as r:
            print(r.status, r.headers.get("Content-Length"),
                  r.headers.get("Content-Range"), r.headers.get("Accept-Ranges"))
        return
    args.dest.parent.mkdir(parents=True, exist_ok=True)
    if args.dest.exists():
        print(f"already present: {args.dest}")
        return
    download(url, args.dest)


if __name__ == "__main__":
    main()
