"""Download actual weights and verify every exported file; run on the client."""

import argparse
import json
from pathlib import Path

from compute.client import ComputeClient

from export import unpack_archive, unpack_result


def retrieve(run_id: str, archive: Path | None, output: Path):
    client = ComputeClient()
    run = client.get_run(run_id)
    if run["status"] != "succeeded":
        raise RuntimeError(f"Run is {run['status']}; expected a successful export")
    result = client.download_result(run)
    if archive is None:
        if result.get("export", {}).get("format") != "tar.gz+base64":
            raise ValueError("This run needs --archive from its scoped transfer")
        verified = unpack_result(result, output)
    else:
        verified = unpack_archive(archive.read_bytes(), result["export"]["sha256"], output)
    receipt = client.get_receipt(run_id)
    (output / "compute-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    summary = {k: v for k, v in result.items() if k != "export"}
    summary.update({"run_id": run_id, "export_verified": verified})
    (output / "download.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_id")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--archive", type=Path, help="Only needed for older scoped-transfer runs")
    args = parser.parse_args()
    retrieve(args.run_id, args.archive, args.out)
