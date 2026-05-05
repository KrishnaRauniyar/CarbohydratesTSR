#!/usr/bin/env python3
"""Filter sample-detail rows to PDB IDs with available legacy PDB files."""

from __future__ import annotations

import argparse
import csv
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence
from urllib import error, request


PDB_DOWNLOAD_URL = "https://files.rcsb.org/download/{pdb_id}.pdb"
LEGACY_PDB_MARKERS = (
    "HEADER",
    "TITLE ",
    "COMPND",
    "SOURCE",
    "KEYWDS",
    "EXPDTA",
    "AUTHOR",
    "REVDAT",
    "JRNL  ",
    "REMARK",
    "SEQRES",
    "ATOM  ",
    "HETATM",
    "TER   ",
    "END   ",
)


@dataclass(frozen=True)
class LegacyCheckResult:
    pdb_id: str
    has_legacy_pdb: bool
    status: str
    pdb_path: str
    message: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read a sample-details CSV and keep only rows whose protein/PDB ID "
            "has an available legacy .pdb file."
        )
    )
    parser.add_argument(
        "-i",
        "--input-csv",
        required=True,
        type=Path,
        help="Input sample-details CSV.",
    )
    parser.add_argument(
        "-o",
        "--output-csv",
        required=True,
        type=Path,
        help="Filtered CSV containing only rows with confirmed legacy PDB files.",
    )
    parser.add_argument(
        "--removed-csv",
        type=Path,
        help="Optional CSV for rows removed from the input.",
    )
    parser.add_argument(
        "--report-csv",
        type=Path,
        help="Optional per-PDB check report CSV.",
    )
    parser.add_argument(
        "--download-dir",
        required=True,
        type=Path,
        help="Directory used to read/cache downloaded legacy PDB files.",
    )
    parser.add_argument(
        "--protein-column",
        default="protein",
        help="CSV column containing the PDB ID. Default: protein.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of parallel PDB checks/downloads. Default: 8.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=60,
        help="Download timeout in seconds. Default: 60.",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Download PDB files again even when a cached file exists.",
    )
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="Only use existing cached PDB files; missing cache entries are removed.",
    )
    return parser.parse_args()


def read_csv(path: Path, protein_column: str) -> tuple[list[str], list[dict[str, str]]]:
    if not path.is_file():
        raise FileNotFoundError(f"Input CSV does not exist: {path}")

    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Input CSV is empty: {path}")
        if protein_column not in reader.fieldnames:
            raise ValueError(
                f"Column '{protein_column}' not found. Available columns: {', '.join(reader.fieldnames)}"
            )
        return reader.fieldnames, list(reader)


def legacy_pdb_file_looks_valid(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size == 0:
        return False

    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for _ in range(200):
            line = handle.readline()
            if not line:
                break
            if line.startswith(LEGACY_PDB_MARKERS):
                return True
            if "PDBx/mmCIF" in line or line.startswith("data_"):
                return False
    return False


def find_cached_pdb(pdb_id: str, download_dir: Path) -> Path | None:
    candidates = [
        download_dir / f"{pdb_id}.pdb",
        download_dir / f"{pdb_id.upper()}.pdb",
        download_dir / f"{pdb_id.lower()}.pdb",
    ]
    for candidate in candidates:
        if legacy_pdb_file_looks_valid(candidate):
            return candidate
    return None


def download_pdb(pdb_id: str, output_path: Path, timeout: int) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    url = PDB_DOWNLOAD_URL.format(pdb_id=pdb_id.upper())
    req = request.Request(url, headers={"User-Agent": "CarbohydratesTSR/1.0"})
    with request.urlopen(req, timeout=timeout) as response:
        output_path.write_bytes(response.read())


def check_pdb_id(
    pdb_id: str,
    download_dir: Path,
    timeout: int,
    force_download: bool,
    no_download: bool,
) -> LegacyCheckResult:
    normalized_pdb_id = pdb_id.strip().upper()
    if not normalized_pdb_id:
        return LegacyCheckResult(pdb_id, False, "empty_pdb_id", "", "Empty PDB ID")

    cached_path = None if force_download else find_cached_pdb(normalized_pdb_id, download_dir)
    if cached_path is not None:
        return LegacyCheckResult(
            normalized_pdb_id,
            True,
            "cached_legacy_pdb",
            str(cached_path),
            "Found valid cached legacy PDB file",
        )

    output_path = download_dir / f"{normalized_pdb_id}.pdb"
    if no_download:
        return LegacyCheckResult(
            normalized_pdb_id,
            False,
            "missing_cached_pdb",
            str(output_path),
            "No valid cached legacy PDB file found and downloads are disabled",
        )

    try:
        download_pdb(normalized_pdb_id, output_path, timeout)
    except error.HTTPError as exc:
        if exc.code in {400, 404}:
            return LegacyCheckResult(
                normalized_pdb_id,
                False,
                "legacy_pdb_unavailable",
                str(output_path),
                f"RCSB legacy .pdb download returned HTTP {exc.code}",
            )
        return LegacyCheckResult(
            normalized_pdb_id,
            False,
            "download_http_error",
            str(output_path),
            f"RCSB legacy .pdb download failed with HTTP {exc.code}: {exc.reason}",
        )
    except Exception as exc:  # noqa: BLE001
        return LegacyCheckResult(
            normalized_pdb_id,
            False,
            "download_error",
            str(output_path),
            str(exc),
        )

    if legacy_pdb_file_looks_valid(output_path):
        return LegacyCheckResult(
            normalized_pdb_id,
            True,
            "downloaded_legacy_pdb",
            str(output_path),
            "Downloaded valid legacy PDB file",
        )

    return LegacyCheckResult(
        normalized_pdb_id,
        False,
        "invalid_downloaded_pdb",
        str(output_path),
        "Downloaded file does not look like a legacy PDB file",
    )


def write_rows(path: Path, fieldnames: Sequence[str], rows: Sequence[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_report(path: Path, results: Sequence[LegacyCheckResult]) -> None:
    fieldnames = ["pdb_id", "has_legacy_pdb", "status", "pdb_path", "message"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for result in sorted(results, key=lambda item: item.pdb_id):
            writer.writerow(
                {
                    "pdb_id": result.pdb_id,
                    "has_legacy_pdb": int(result.has_legacy_pdb),
                    "status": result.status,
                    "pdb_path": result.pdb_path,
                    "message": result.message,
                }
            )


def main() -> int:
    args = parse_args()

    fieldnames, rows = read_csv(args.input_csv, args.protein_column)
    pdb_ids = sorted({row[args.protein_column].strip().upper() for row in rows if row[args.protein_column].strip()})
    if not pdb_ids:
        print("[ERROR] No PDB IDs found in input CSV.", file=sys.stderr)
        return 1

    print(f"[INFO] Input rows: {len(rows)}")
    print(f"[INFO] Unique PDB IDs to check: {len(pdb_ids)}")
    print(f"[INFO] Download/cache dir: {args.download_dir}")

    results: list[LegacyCheckResult] = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = {
            executor.submit(
                check_pdb_id,
                pdb_id,
                args.download_dir,
                args.timeout,
                args.force_download,
                args.no_download,
            ): pdb_id
            for pdb_id in pdb_ids
        }
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            if result.has_legacy_pdb:
                print(f"[INFO] KEEP {result.pdb_id}: {result.status}")
            else:
                print(f"[WARN] REMOVE {result.pdb_id}: {result.status} - {result.message}", file=sys.stderr)

    keep_pdb_ids = {result.pdb_id for result in results if result.has_legacy_pdb}
    kept_rows = [
        row for row in rows if row[args.protein_column].strip().upper() in keep_pdb_ids
    ]
    removed_rows = [
        row for row in rows if row[args.protein_column].strip().upper() not in keep_pdb_ids
    ]

    write_rows(args.output_csv, fieldnames, kept_rows)
    if args.removed_csv:
        write_rows(args.removed_csv, fieldnames, removed_rows)
    if args.report_csv:
        write_report(args.report_csv, results)

    removed_pdb_ids = len(pdb_ids) - len(keep_pdb_ids)
    print(f"[INFO] Kept PDB IDs: {len(keep_pdb_ids)}")
    print(f"[INFO] Removed PDB IDs: {removed_pdb_ids}")
    print(f"[INFO] Kept rows: {len(kept_rows)}")
    print(f"[INFO] Removed rows: {len(removed_rows)}")
    print(f"[INFO] Wrote filtered CSV: {args.output_csv}")
    if args.removed_csv:
        print(f"[INFO] Wrote removed rows CSV: {args.removed_csv}")
    if args.report_csv:
        print(f"[INFO] Wrote report CSV: {args.report_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
