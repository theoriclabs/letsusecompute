"""Budget-bound repair evaluation through the existing Compute service only."""

import argparse
import base64
import hashlib
import json
from dataclasses import replace
from pathlib import Path

from compute.ast import plan_payload
from compute.ast.archive import pack_payload
from compute.ast.models import SourceFile
from compute.cli.http_run import _launch_fields, _machine_for_quote, _quote_body
from compute.cli.preflight import build_preflight_from_quote
from compute.client import ComputeClient, is_terminal_status
from compute.launch import resolve_launch_selection

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--archive", type=Path, required=True)
parser.add_argument("--training-run", required=True)
parser.add_argument("--ledger", type=Path, required=True)
parser.add_argument("--gpu", default="runpod/H100-SXM")
parser.add_argument("--timeout", type=int, default=5400)
parser.add_argument("--plan-only", action="store_true")
args = parser.parse_args()
root = Path(__file__).resolve().parent
raw = args.archive.read_bytes()
digest = hashlib.sha256(raw).hexdigest()
plan = plan_payload(str(root / "train.py") + "::repair", project_root=root)
assert "embedded_training.py" in plan.files
embedded = ("ARCHIVE_B64 = " + repr(base64.b64encode(raw).decode("ascii")) + "\n").encode()
plan = replace(
    plan,
    source_files=tuple(
        SourceFile(p.path, embedded) if p.path == "embedded_training.py" else p
        for p in plan.source_files
    ),
)
archive = pack_payload(plan)
print(
    json.dumps(
        {
            "source_files": list(plan.files),
            "payload_bytes": len(archive),
            "training_archive_sha256": digest,
            "public_endpoint": False,
        }
    ),
    flush=True,
)
if args.plan_only:
    raise SystemExit(0)
ledger_path = args.ledger.resolve()
ledger = json.loads(ledger_path.read_text())
client = ComputeClient()
spent = 0
for run_id in ledger["run_ids"]:
    run = client.get_run(run_id)
    if not is_terminal_status(run["status"]):
        raise RuntimeError("An experiment run is active: " + run_id)
    spent += client.get_receipt(run_id)["total_debit_cents"]
if client.list_machines():
    raise RuntimeError("An account machine is active")
remaining = round(ledger["budget_usd"] * 100) - spent
training = client.get_run(args.training_run)
if (
    training["status"] != "succeeded"
    or client.download_result(training)["export"]["sha256"] != digest
):
    raise ValueError("Local archive differs from the named successful training run")
selection = resolve_launch_selection(args.gpu, None, plan.function_config.gpu_sku, None)
fields = _launch_fields(
    plan,
    selection=selection,
    timeout=args.timeout,
    args={"training_sha256": digest, "training_run_id": args.training_run, "count": 16},
)
payload = client.push_payload(archive=archive, manifest=plan.manifest.to_dict())
quote = client.create_run_quote(
    _quote_body(
        plan,
        fields=fields,
        payload_id=payload["payload_id"],
        machine=_machine_for_quote(client, selection),
        args=fields["args_json"],
    )
)
preflight = build_preflight_from_quote(client, quote=quote, timeout_seconds=args.timeout)
if quote["hold"]["max_hold_cents"] > remaining or preflight.estimated_total_cents > remaining:
    raise RuntimeError("Live quote exceeds remaining authorized budget")
print(
    json.dumps({"remaining_cents": remaining, "quote": quote, "preflight": preflight.to_dict()}),
    flush=True,
)
run = client.create_run(
    {**fields, "payload_id": payload["payload_id"], "quote_id": quote["quote_id"]}
)
record = ledger_path.parent / run["id"]
record.mkdir()
source = record / "source"
source.mkdir()
for item in plan.source_files:
    # Keep the actual private payload under ignored weights/, not in blog source.
    data = (root / item.path).read_bytes() if item.path == "embedded_training.py" else item.content
    (source / item.path).write_bytes(data)
payload_path = root / "weights" / (run["id"] + "-payload.tar.zst")
payload_path.write_bytes(archive)
(record / "launch.json").write_text(
    json.dumps(
        {
            "run_id": run["id"],
            "stage": "repair",
            "quote": quote,
            "preflight": preflight.to_dict(),
            "source_files": list(plan.files),
            "payload_sha256": hashlib.sha256(archive).hexdigest(),
            "payload_path": str(payload_path),
            "training_archive_sha256": digest,
            "source_note": "embedded_training.py is a placeholder here; exact embedded data is retained in the ignored payload archive.",
            "transport": "native Compute payload/result; no public tunnel",
        },
        indent=2,
    )
    + "\n"
)
ledger["run_ids"].append(run["id"])
ledger.update(
    gpu_api_spend_usd=spent / 100,
    remaining_budget_usd=remaining / 100,
    launch_status="repair running",
    all_experiment_gpus_terminated=False,
    temporary_transfer_stopped=True,
)
ledger_path.write_text(json.dumps(ledger, indent=2) + "\n")
print(json.dumps({"run_id": run["id"], "status": run["status"]}), flush=True)
