import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional


@dataclass
class SummaryRow:
    lambda_coef: str
    dict_size: int
    avg_l0: float
    mse: float
    l1: float
    r2: float
    log_path: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize SAE training logs into JSON and Markdown tables."
    )
    parser.add_argument("--logs_dir", default="logs")
    parser.add_argument("--epoch", type=int, default=5)
    parser.add_argument(
        "--out_json", default="artifacts/metrics/sae_log_summary.json"
    )
    parser.add_argument(
        "--out_md", default="artifacts/metrics/sae_log_summary.md"
    )
    return parser.parse_args()


def iter_train_logs(logs_dir: Path) -> Iterable[Path]:
    yield from logs_dir.glob("sae_*/**/train.jsonl")


def parse_lambda_from_path(path: Path) -> str:
    name = path.parent.name
    if "_l3e2" in name:
        return "3e-2"
    if "_l1e2" in name:
        return "1e-2"
    if re.search(r"_l1(_|$)", name):
        return "1e-3"
    match = re.search(r"_l(\d+e\d+)", name)
    if match:
        coef = match.group(1)
        return coef.replace("e", "e-")
    return "unknown"


def last_epoch_record(log_path: Path, epoch: int) -> Optional[Dict[str, float]]:
    last: Optional[Dict[str, float]] = None
    with log_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            if int(record.get("epoch", -1)) == epoch:
                last = record
    return last


def summarize(logs_dir: Path, epoch: int) -> List[SummaryRow]:
    rows_by_lambda: Dict[str, SummaryRow] = {}
    for log_path in sorted(iter_train_logs(logs_dir)):
        record = last_epoch_record(log_path, epoch)
        if record is None:
            continue
        lambda_coef = parse_lambda_from_path(log_path)
        if lambda_coef in rows_by_lambda:
            continue
        rows_by_lambda[lambda_coef] = SummaryRow(
            lambda_coef=lambda_coef,
            dict_size=int(record["dict_size"]),
            avg_l0=float(record["l0"]),
            mse=float(record["mse"]),
            l1=float(record["l1"]),
            r2=float(record["r2"]),
            log_path=str(log_path),
        )
    rows = list(rows_by_lambda.values())
    rows.sort(key=lambda r: r.lambda_coef)
    return rows


def write_json(rows: List[SummaryRow], out_json: Path, epoch: int) -> None:
    out_json.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "epoch": epoch,
        "rows": [
            {
                "lambda": r.lambda_coef,
                "dict_size": r.dict_size,
                "avg_l0": r.avg_l0,
                "mse": r.mse,
                "l1": r.l1,
                "r2": r.r2,
                "log_path": r.log_path,
            }
            for r in rows
        ],
    }
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def write_md(rows: List[SummaryRow], out_md: Path) -> str:
    out_md.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "| L1 coefficient (lambda) | Dictionary size | Avg. L0 | Reconstruction MSE | L1 | R2 |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for r in rows:
        lines.append(
            "| {lam} | {dict_size} | {l0:.2f} | {mse:.6g} | {l1:.6g} | {r2:.6g} |".format(
                lam=r.lambda_coef,
                dict_size=r.dict_size,
                l0=r.avg_l0,
                mse=r.mse,
                l1=r.l1,
                r2=r.r2,
            )
        )
    content = "\n".join(lines) + "\n"
    out_md.write_text(content, encoding="utf-8")
    return content


def main() -> None:
    args = parse_args()
    logs_dir = Path(args.logs_dir)
    rows = summarize(logs_dir, args.epoch)
    write_json(rows, Path(args.out_json), args.epoch)
    md = write_md(rows, Path(args.out_md))
    print(md)


if __name__ == "__main__":
    main()
