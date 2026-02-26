"""Clear terminal verifier for state/action, cameras, and text mappings."""
import argparse, json, sys
from pathlib import Path
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

def _load_json(path: Path) -> dict: return json.loads(path.read_text(encoding="utf-8"))
def _load_tasks(meta_dir: Path) -> tuple[str, str] | None:
    tasks_json = meta_dir / "tasks.json"
    if tasks_json.exists():
        tasks = json.loads(tasks_json.read_text(encoding="utf-8"))
        return tasks_json.name, json.dumps(tasks, indent=2, ensure_ascii=False)

    tasks_jsonl = meta_dir / "tasks.jsonl"
    if tasks_jsonl.exists():
        tasks = []
        for line in tasks_jsonl.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            tasks.append(json.loads(line))
        return tasks_jsonl.name, json.dumps(tasks, indent=2, ensure_ascii=False)

    return None
def _feature_dim(info: dict, key: str) -> int | None:
    shape = info.get("features", {}).get(key, {}).get("shape")
    return int(shape[0]) if isinstance(shape, list) and shape else None
def _parse_section(modality: dict, section: str) -> tuple[list[tuple[str, int, int]], int, list[str]]:
    items = sorted(modality.get(section, {}).items(), key=lambda kv: kv[1].get("start", -1))
    if not items: return [], 0, [f"Missing or empty section: {section}"]
    parsed, errs, prev_end = [], [], 0
    for key, rng in items:
        s, e = int(rng.get("start", -1)), int(rng.get("end", -1))
        if s < 0 or e <= s: errs.append(f"{section}.{key} invalid range [{s}, {e})")
        if s < prev_end: errs.append(f"{section}.{key} overlaps previous range")
        prev_end = max(prev_end, e); parsed.append((key, s, e))
    return parsed, prev_end, errs
def _length_stats(col: pa.ChunkedArray, exp: int | None) -> tuple[int, int, int, int, list[tuple[int, int]]]:
    mn, mx, row0, bad, ex = sys.maxsize, 0, 0, 0, []
    for chunk in col.chunks:
        lens = np.full(len(chunk), chunk.type.list_size, np.int64) if pa.types.is_fixed_size_list(chunk.type) else np.diff(chunk.offsets.to_numpy(zero_copy_only=False).astype(np.int64))
        if lens.size:
            mn, mx = min(mn, int(lens.min())), max(mx, int(lens.max()))
            if exp is not None:
                idx = np.where(lens != exp)[0]; bad += int(idx.size)
                for i in idx[: max(0, 3 - len(ex))]: ex.append((row0 + int(i), int(lens[i])))
        row0 += len(lens)
    return (0 if mn == sys.maxsize else mn), mx, row0, bad, ex
def _mappings(modality: dict, section: str) -> list[tuple[str, str]]:
    return [(n, c.get("original_key", n) if isinstance(c, dict) else n) for n, c in modality.get(section, {}).items()]
def _resolve(section: str, key: str, feats: set[str]) -> str | None:
    if key in feats: return key
    if section == "annotation" and f"annotation.{key}" in feats: return f"annotation.{key}"
    if section == "video" and f"observation.images.{key}" in feats: return f"observation.images.{key}"
    return None

def main() -> None:
    p = argparse.ArgumentParser(description="Verify modality slices and parquet state/action dimensions.")
    p.add_argument("--dataset-path", type=Path, required=True); p.add_argument("--max-files", type=int, default=None)
    args = p.parse_args(); root = args.dataset_path.expanduser().resolve()
    modality, info = _load_json(root / "meta/modality.json"), _load_json(root / "meta/info.json")
    state, state_dim, errs = _parse_section(modality, "state")
    action, action_dim, act_errs = _parse_section(modality, "action"); errs += act_errs
    files = sorted(root.glob("data/chunk-*/episode_*.parquet")) or sorted(root.glob("data/chunk-*/file-*.parquet"))
    files = files[: args.max_files] if args.max_files is not None else files
    if not files: raise FileNotFoundError(f"No parquet files under {root / 'data'}")
    feats, pq_cols, warns = set(info.get("features", {}).keys()), set(pq.read_schema(files[0]).names), []
    task_payload = None
    try:
        task_payload = _load_tasks(root / "meta")
    except (json.JSONDecodeError, OSError, UnicodeDecodeError) as e:
        warns.append(f"failed to load tasks metadata: {e}")
    print("=" * 72); print("LeRobot state/action verification"); print(f"Dataset: {root}")
    print("[Modality ranges]")
    for name, sec, dim in [("state", state, state_dim), ("action", action, action_dim)]:
        print(f"  {name} ({len(sec)} keys, dim={dim})")
        for k, s, e in sec: print(f"    - {k:<20} [{s:>3}:{e:<3}] width={e - s}")
    print("[Cameras]")
    cams = _mappings(modality, "video")
    if not cams: print("  none")
    for n, k in cams:
        r = _resolve("video", k, feats); ii, ip = r is not None, (r in pq_cols if r else False)
        if not ii: warns.append(f"camera '{n}' key not found in info.features: {k}")
        print(f"  - {n:<20} -> {(r or k):<32} info={'yes' if ii else 'no '} parquet={'yes' if ip else 'no '}")
    print("[Text/annotation]")
    txt = _mappings(modality, "annotation")
    if not txt: print("  none")
    for n, k in txt:
        r = _resolve("annotation", k, feats); ii, ip = r is not None, (r in pq_cols if r else False)
        if not ii: warns.append(f"text '{n}' key not found in info.features: {k}")
        print(f"  - {n:<20} -> {(r or k):<32} info={'yes' if ii else 'no '} parquet={'yes' if ip else 'no '}")
    print("[Tasks]")
    if task_payload is None:
        print("  none (missing meta/tasks.json or meta/tasks.jsonl)")
    else:
        task_file, task_content = task_payload
        print(f"  source: {task_file}")
        print(task_content)
    isd, iad = _feature_dim(info, "observation.state"), _feature_dim(info, "action")
    if isd is not None and isd != state_dim: errs.append(f"state dim mismatch modality={state_dim} info={isd}")
    if iad is not None and iad != action_dim: errs.append(f"action dim mismatch modality={action_dim} info={iad}")
    es, ea = (isd if isd is not None else state_dim), (iad if iad is not None else action_dim)
    print(f"[Dimension checks]\n  observation.state expected={es} (modality={state_dim}, info={isd})")
    print(f"  action            expected={ea} (modality={action_dim}, info={iad})")
    rows = sbad = abad = 0; smin = amin = sys.maxsize; smax = amax = 0; samples = []
    for f in files:
        t = pq.read_table(f, columns=["observation.state", "action"])
        smn, smx, nrows, s_bad, s_ex = _length_stats(t.column("observation.state"), es)
        amn, amx, _n, a_bad, a_ex = _length_stats(t.column("action"), ea)
        rows += nrows; smin, smax, amin, amax = min(smin, smn), max(smax, smx), min(amin, amn), max(amax, amx)
        sbad, abad = sbad + s_bad, abad + a_bad; samples += [(f.name, "state", r, l) for r, l in s_ex] + [(f.name, "action", r, l) for r, l in a_ex]
    smin, amin = (0 if smin == sys.maxsize else smin), (0 if amin == sys.maxsize else amin)
    print("[Parquet scan]"); print(f"  files scanned: {len(files)}"); print(f"  rows scanned : {rows}")
    print(f"  state length : [{smin}, {smax}] expected={es} mismatches={sbad}")
    print(f"  action length: [{amin}, {amax}] expected={ea} mismatches={abad}")
    if sbad: errs.append(f"state rows with wrong length: {sbad} (expected {es})")
    if abad: errs.append(f"action rows with wrong length: {abad} (expected {ea})")
    if samples: print("[Mismatch examples]"); [print(f"  - file={n:<20} col={k:<6} row={r:<6} len={l}") for n, k, r, l in samples[:6]]
    if warns: print("[Warnings]"); [print(f"  - {w}") for w in warns]
    if errs: print("[Result] FAIL"); [print(f"  - {e}") for e in errs]; raise SystemExit(1)
    print("[Result] PASS")

if __name__ == "__main__": main()
