"""
Regenerates both systems' 105 reports under seed 7 (the primary campaign
ran seed 42) into reports_metropt_B8S7.parquet / reports_metropt_B3RS7.parquet,
then L1 + L2/L3 judging follow via the standard scripts once SYSTEMS is
extended. Everything deterministic (handler verdicts, digests, facts) is
identical; only the LLM decoding seed changes, isolating seed sensitivity
of the small error counts (the 4 errors behind 3.9%; B3R's 43 false
fault claims).

Usage: python metropt_seed7.py [--workers 4]
"""
from __future__ import annotations

import argparse
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

import metropt_campaign as mc
import metropt_b3r_campaign as b3r
from metropt_handlers import handle, load_spec
from metropt_l1 import _feat_row

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs" / "metropt3"
SEED = 7

mc.SEED = SEED        # module-level seed consumed by gen_b3/gen_b5/gen_b8
b3r.SEED = SEED       # seed consumed by gen_b3r


def gen_row(sys_id: str, gen, feat, verdict, spec, vehicle):
    if sys_id == "B8S7":
        out = gen(feat, verdict, spec, vehicle)
    else:
        out = gen(feat, vehicle, spec)
    out["system_id"] = sys_id
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    spec = load_spec()
    rep0 = pd.read_parquet(RUNS / "reports_metropt_B1.parquet")
    plans = [("B8S7", mc.gen_b8), ("B3RS7", b3r.gen_b3r)]

    for sys_id, gen in plans:
        out = RUNS / f"reports_metropt_{sys_id}.parquet"
        done = set()
        if out.exists():
            old = pd.read_parquet(out)
            done = {(r.vehicle, r.window_id) for r in old.itertuples()
                    if r.ok and (r.text or "").strip()}
        tasks = [(r.vehicle, r.window_id, r.kind, r.offset_h, r.event_id)
                 for _, r in rep0.iterrows()
                 if (r.vehicle, r.window_id) not in done]
        print(f"{sys_id}: {len(tasks)} to generate ({len(done)} done)")
        if not tasks:
            continue
        lock, rows = threading.Lock(), []

        def _save(new: pd.DataFrame) -> None:
            full = new if not out.exists() else pd.concat(
                [pd.read_parquet(out), new], ignore_index=True)
            full = full.drop_duplicates(subset=["vehicle", "window_id"],
                                        keep="last")
            full.to_parquet(out, index=False)

        def work(t):
            veh, wid, kind, off, ev = t
            feat = _feat_row(veh, wid)
            verdict = handle(feat, veh, spec)
            g = gen_row(sys_id, gen, feat, verdict, spec, veh)
            return {"vehicle": veh, "window_id": wid, "kind": kind,
                    "offset_h": off, "event_id": ev, **g}

        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(work, t): t for t in tasks}
            persisted = 0
            for i, fut in enumerate(as_completed(futs), 1):
                try:
                    row = fut.result()
                except Exception as e:  # noqa: BLE001
                    t = futs[fut]
                    row = {"vehicle": t[0], "window_id": t[1], "kind": t[2],
                           "offset_h": t[3], "event_id": t[4], "ok": False,
                           "error": str(e)[:200]}
                with lock:
                    rows.append(row)
                    if i % 20 == 0 or i == len(tasks):
                        _save(pd.DataFrame(rows[persisted:]))
                        persisted = len(rows)
                        print(f"  {sys_id}: {i}/{len(tasks)}")
        new = pd.DataFrame(rows[persisted:])
        if len(new):
            _save(new)
        n = pd.read_parquet(out)
        print(f"{sys_id} complete: {len(n)} rows, {int(n.ok.sum())} ok")


if __name__ == "__main__":
    main()
