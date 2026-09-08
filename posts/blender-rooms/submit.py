"""Submit a stage with a live quote and a receipt-based experiment budget.

Use the interpreter containing the Compute SDK. No token values are printed.
"""

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from urllib.parse import urlsplit

from compute.ast import plan_payload
from compute.ast.archive import pack_payload
from compute.cli.http_run import _launch_fields, _machine_for_quote, _quote_body
from compute.cli.preflight import build_preflight_from_quote
from compute.client import ComputeClient, is_terminal_status
from compute.launch import resolve_launch_selection

parser = argparse.ArgumentParser()
parser.add_argument("stage", choices=["smoke", "train", "test", "control"])
parser.add_argument("--timeout", type=int, required=True)
parser.add_argument("--count", type=int, default=32)
parser.add_argument("--gpu", default="runpod/A100-PCIe-80GB")
parser.add_argument("--training-run")
parser.add_argument("--transfer-dir", type=Path, required=True)
parser.add_argument("--ledger", type=Path, required=True)
parser.add_argument("--budget", type=float, help="Required only when creating a new budget ledger")
parser.add_argument(
    "--plan-only", action="store_true", help="Resolve and pack source without network or spending"
)
args = parser.parse_args()
root = Path(__file__).resolve().parent
ledger_path = args.ledger.resolve()
if args.plan_only:
    plan = plan_payload(str(root / "train.py") + "::" + args.stage, project_root=root)
    print(
        json.dumps(
            {
                "stage": args.stage,
                "source_files": list(plan.files),
                "archive_bytes": len(pack_payload(plan)),
            },
            indent=2,
        )
    )
    raise SystemExit(0)
if ledger_path.exists():
    ledger = json.loads(ledger_path.read_text())
    if args.budget is not None and round(args.budget * 100) != round(ledger["budget_usd"] * 100):
        raise ValueError("Existing ledger budget differs; use its original cap")
else:
    if args.budget is None or not 0 < args.budget < 100000:
        raise ValueError("A new ledger requires an explicit positive --budget in USD")
    ledger = {"budget_usd": round(args.budget, 2), "run_ids": []}
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    ledger_path.write_text(json.dumps(ledger, indent=2) + "\n")
client = ComputeClient()
spent = 0
for run_id in ledger["run_ids"]:
    run = client.get_run(run_id)
    if not is_terminal_status(run["status"]):
        raise RuntimeError("An experiment run is still active: " + run_id)
    receipt = client.get_receipt(run_id)
    spent += receipt["total_debit_cents"]
if client.list_machines():
    raise RuntimeError("An account machine is active; refusing an overlapping allocation")
remaining = round(ledger["budget_usd"] * 100) - spent
transfer = args.transfer_dir.resolve()
base = transfer.joinpath("base-url").read_text().strip().rstrip("/")
parsed = urlsplit(base)
if (
    parsed.scheme != "https"
    or not parsed.hostname
    or parsed.username
    or parsed.query
    or parsed.fragment
):
    raise ValueError("The transfer base must be a plain HTTPS URL")
slot = {"smoke": "smoke", "train": "sft", "test": "test", "control": "probe"}[args.stage]
if (transfer / "received" / (slot + ".tar.gz")).exists():
    raise RuntimeError("Transfer slot already complete; preserve it before launching a new run")
kwargs = {
    "export_url": base + "/upload/" + slot,
    "export_token": (transfer / "worker.key").read_text().strip(),
}
if args.stage in {"test", "control"}:
    saved = json.loads((transfer / "received/sft.json").read_text())
    if not args.training_run:
        raise ValueError("training run is required")
    training = client.get_run(args.training_run)
    if training["status"] != "succeeded":
        raise ValueError("Training run must have succeeded")
    if client.download_result(training)["export"]["sha256"] != saved["sha256"]:
        raise ValueError("Transferred archive does not match the named training run")
    kwargs.update(
        training_archive_url=base + "/download/sft",
        training_sha256=saved["sha256"],
        training_run_id=args.training_run,
        count=args.count,
    )
plan = plan_payload(str(root / "train.py") + "::" + args.stage, project_root=root)
selection = resolve_launch_selection(args.gpu, None, plan.function_config.gpu_sku, None)
fields = _launch_fields(plan, selection=selection, timeout=args.timeout, args=kwargs)
archive = pack_payload(plan)
payload = client.push_payload(archive=archive, manifest=plan.manifest.to_dict())
quote = client.create_run_quote(
    _quote_body(
        plan,
        fields=fields,
        payload_id=payload["payload_id"],
        machine=_machine_for_quote(client, selection),
        args=kwargs,
    )
)
preflight = build_preflight_from_quote(client, quote=quote, timeout_seconds=args.timeout)
hold = quote["hold"]["max_hold_cents"]
if hold > remaining or preflight.estimated_total_cents > remaining:
    raise RuntimeError("Live quote exceeds remaining authorized budget")
print(
    json.dumps(
        {"remaining_cents": remaining, "quote": quote, "preflight": preflight.to_dict()}, indent=2
    ),
    flush=True,
)
run = client.create_run(
    {**fields, "payload_id": payload["payload_id"], "quote_id": quote["quote_id"]}
)
record = ledger_path.parent / run["id"]
record.mkdir()
(record / "payload.tar.zst").write_bytes(archive)
source = record / "source"
source.mkdir()
for name in plan.files:
    target = source / name
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(root / name, target)
(record / "launch.json").write_text(
    json.dumps(
        {
            "run_id": run["id"],
            "stage": args.stage,
            "quote": quote,
            "preflight": preflight.to_dict(),
            "source_files": list(plan.files),
            "payload_sha256": hashlib.sha256(archive).hexdigest(),
        },
        indent=2,
    )
    + "\n"
)
ledger["run_ids"].append(run["id"])
ledger.update(
    gpu_api_spend_usd=spent / 100,
    remaining_budget_usd=remaining / 100,
    launch_status=args.stage + " running",
    all_experiment_gpus_terminated=False,
    temporary_transfer_stopped=False,
)
ledger_path.write_text(json.dumps(ledger, indent=2) + "\n")
print(json.dumps({"run_id": run["id"], "stage": args.stage, "status": run["status"]}), flush=True)
