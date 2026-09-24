"""Teach Qwen3-0.6B to play Pokémon Showdown random battles from text.

One job does everything on a fresh machine:

1. installs Node and a pinned Pokémon Showdown, and starts a local server;
2. plays scripted-expert battles (poke-env's SimpleHeuristicsPlayer, tera off)
   and records each decision as (battle state as text, option number);
3. LoRA fine-tunes Qwen3-0.6B to answer with that option number;
4. plays every player 200 battles against poke-env's MaxBasePowerPlayer and
   reports win rates: random, untuned Qwen, fine-tuned Qwen, the expert.

The model sees numbered options (moves, then switches) and answers one digit.
Both Qwen players pick the highest-probability legal digit, so the untuned
model gets exactly the same constrained decoding as the trained one.
"""
from __future__ import annotations

import asyncio
import json
import os
import random
import shutil
import socket
import subprocess
import sys
import tarfile
import time
import urllib.request
from pathlib import Path

import compute

app = compute.App("qwen-pokemon")
image = compute.Image.cuda_pytorch().pip_install(
    "poke-env==0.16.1", "transformers==4.51.3", "peft==0.15.2",
    "accelerate==1.6.0", "huggingface_hub==0.30.2")
hf_secret = compute.Secret.from_name("hf")

BASE = "Qwen/Qwen3-0.6B"
REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"
FORMAT = "gen9randombattle"
NODE_VERSION = "v22.22.0"
SHOWDOWN_COMMIT = "a5df8274e85b0889bf2a9b3422a08b39732374fc"
HUB_REPO = "theoriclabs/qwen3-0.6b-pokemon-showdown"
PORT = 8000


# --------------------------------------------------------------------------- #
# Showdown server, installed and started inside the job
# --------------------------------------------------------------------------- #

def start_showdown(work: Path) -> tuple[subprocess.Popen, dict]:
    """Download Node + Showdown, build, start on localhost. Returns timings."""
    timings = {}
    t = time.time()
    node_dir = work / f"node-{NODE_VERSION}-linux-x64"
    if shutil.which("node") and not os.environ.get("POKEMON_FORCE_NODE"):
        node_bin = Path(shutil.which("node")).parent
    else:
        archive = work / "node.tar.xz"
        urllib.request.urlretrieve(
            f"https://nodejs.org/dist/{NODE_VERSION}/node-{NODE_VERSION}-linux-x64.tar.xz", archive)
        with tarfile.open(archive) as tar:
            tar.extractall(work)
        node_bin = node_dir / "bin"
    env = dict(os.environ, PATH=f"{node_bin}:{os.environ['PATH']}")
    timings["node_seconds"] = time.time() - t

    t = time.time()
    ps = work / f"pokemon-showdown-{SHOWDOWN_COMMIT}"
    try:
        if (ps / "package.json").exists():
            raise FileExistsError  # reuse a previous checkout (local re-runs)
        archive = work / "showdown.tar.gz"
        urllib.request.urlretrieve(
            f"https://github.com/smogon/pokemon-showdown/archive/{SHOWDOWN_COMMIT}.tar.gz", archive)
        with tarfile.open(archive) as tar:
            tar.extractall(work)
    except FileExistsError:
        pass
    except OSError as e:  # some networks block GitHub archive downloads but allow git
        print("archive download failed, falling back to git:", e, flush=True)
        subprocess.run(["git", "clone", "-q", "https://github.com/smogon/pokemon-showdown.git", str(ps)], check=True)
        subprocess.run(["git", "checkout", "-q", SHOWDOWN_COMMIT], cwd=ps, check=True)
    timings["showdown_download_seconds"] = time.time() - t

    t = time.time()
    subprocess.run(["npm", "install", "--omit=dev", "--no-audit", "--no-fund", "--loglevel=error"],
                   cwd=ps, env=env, check=True)
    timings["npm_install_seconds"] = time.time() - t
    t = time.time()
    subprocess.run(["node", "build"], cwd=ps, env=env, check=True, stdout=subprocess.DEVNULL)
    shutil.copy(ps / "config" / "config-example.js", ps / "config" / "config.js")
    timings["build_seconds"] = time.time() - t

    t = time.time()
    log = open(work / "showdown.log", "w")
    server = subprocess.Popen(["node", "pokemon-showdown", "start", "--no-security", str(PORT)],
                              cwd=ps, env=env, stdout=log, stderr=subprocess.STDOUT)
    for _ in range(240):
        try:
            socket.create_connection(("127.0.0.1", PORT), timeout=1).close()
            break
        except OSError:
            if server.poll() is not None:
                raise RuntimeError("Showdown exited: " + (work / "showdown.log").read_text()[-2000:])
            time.sleep(0.5)
    else:
        raise RuntimeError("Showdown did not open its port in 120 s")
    timings["server_start_seconds"] = time.time() - t
    return server, timings


# --------------------------------------------------------------------------- #
# Battle state as text
# --------------------------------------------------------------------------- #

def _types(mon) -> str:
    return "/".join(t.name.title() for t in mon.types if t is not None)


def _status(mon) -> str:
    bits = [f"{round(100 * mon.current_hp_fraction)}% HP"]
    if mon.status is not None:
        bits.append(mon.status.name.lower())
    boosts = [f"{k.capitalize()} {v:+d}" for k, v in mon.boosts.items() if v]
    if boosts:
        bits.append(", ".join(boosts))
    return ", ".join(bits)


def _mult(x: float) -> str:
    return f"x{x:g}"


def describe(battle) -> tuple[str, list]:
    """Return (prompt text, options). options[i] is a Move or a Pokemon."""
    me, opp = battle.active_pokemon, battle.opponent_active_pokemon
    moves = list(battle.available_moves)
    switches = list(battle.available_switches)
    options = moves + switches
    lines = [f"Pokémon battle ({FORMAT}), turn {battle.turn}."]
    if me is not None:
        s = me.base_stats
        lines.append(f"Your active: {me.species} ({_types(me)}), {_status(me)}. "
                     f"Base Atk {s['atk']}, SpA {s['spa']}, Def {s['def']}, SpD {s['spd']}, Spe {s['spe']}.")
    if opp is not None:
        s = opp.base_stats
        lines.append(f"Opponent active: {opp.species} ({_types(opp)}), {_status(opp)}. "
                     f"Base Atk {s['atk']}, SpA {s['spa']}, Def {s['def']}, SpD {s['spd']}, Spe {s['spe']}.")
    mine_left = sum(not m.fainted for m in battle.team.values())
    theirs_left = 6 - sum(m.fainted for m in battle.opponent_team.values())
    lines.append(f"Pokémon left: you {mine_left}, opponent {theirs_left}.")
    if battle.side_conditions:
        lines.append("Your side: " + ", ".join(c.name.lower() for c in battle.side_conditions) + ".")
    if battle.opponent_side_conditions:
        lines.append("Opponent side: " + ", ".join(c.name.lower() for c in battle.opponent_side_conditions) + ".")
    lines.append("Options:")
    for i, move in enumerate(moves, 1):
        if move.category.name == "STATUS":
            desc = f"{move.type.name.title()}, status"
            if move.boosts and move.target == "self":
                desc += ", raises " + ", ".join(f"{k.capitalize()} {v:+d}" for k, v in move.boosts.items())
        else:
            desc = f"{move.type.name.title()}, {move.category.name.lower()}, power {move.base_power}"
            if me is not None and move.type in me.types:
                desc += " (same type)"
            if opp is not None:
                desc += f", {_mult(opp.damage_multiplier(move))} vs {opp.species}"
        acc = move.accuracy
        if acc is not True and acc < 1:
            desc += f", {round(100 * acc)}% accuracy"
        lines.append(f"{i}. Use {move.id}: {desc}.")
    for i, mon in enumerate(switches, len(moves) + 1):
        desc = f"{_types(mon)}, {_status(mon)}"
        if opp is not None:
            hit = max(opp.damage_multiplier(t) for t in mon.types if t is not None)
            taken = max(mon.damage_multiplier(t) for t in opp.types if t is not None)
            desc += f"; its types hit {opp.species} {_mult(hit)}, {opp.species}'s types hit it {_mult(taken)}"
            desc += f"; base Spe {mon.base_stats['spe']}"
        lines.append(f"{i}. Switch to {mon.species}: {desc}.")
    lines.append("Reply with the number of the best option.")
    return "\n".join(lines), options


def order_index(order, options) -> int | None:
    target = getattr(order, "order", None)
    for i, opt in enumerate(options):
        if opt is target:
            return i
    return None


# --------------------------------------------------------------------------- #
# Players
# --------------------------------------------------------------------------- #

def make_players():
    from poke_env.player import Player, SimpleHeuristicsPlayer

    class Expert(SimpleHeuristicsPlayer):
        """The scripted teacher, with tera turned off so it plays the same game as Qwen."""

        def __init__(self, *a, record=None, **k):
            super().__init__(*a, **k)
            self.record = record

        def choose_move(self, battle):
            order = self.choose_singles_move(battle)[0]
            text, options = describe(battle)
            idx = order_index(order, options)
            if idx is None or len(options) > 9:
                return order if getattr(order, "order", None) is None else self.create_order(order.order)
            if self.record is not None and len(options) > 1:
                self.record.append({"prompt": text, "answer": str(idx + 1), "n_options": len(options),
                                    "switch": idx >= len(battle.available_moves)})
            return self.create_order(options[idx])

    class QwenPlayer(Player):
        """Asks a shared batcher for the argmax legal digit."""

        def __init__(self, *a, batcher=None, **k):
            super().__init__(*a, **k)
            self.batcher = batcher

        async def choose_move(self, battle):
            text, options = describe(battle)
            if not options or len(options) > 9:
                return self.choose_random_move(battle)
            if len(options) == 1:
                return self.create_order(options[0])
            idx = await self.batcher.ask(text, len(options))
            return self.create_order(options[idx])

    return Expert, QwenPlayer


class Batcher:
    """Collects prompts from concurrent battles and scores them in one forward pass."""

    def __init__(self, model, tokenizer, device, max_batch=64):
        self.model, self.tok, self.device, self.max_batch = model, tokenizer, device, max_batch
        self.digit_ids = [tokenizer.convert_tokens_to_ids(str(d)) for d in range(1, 10)]
        self.queue: asyncio.Queue | None = None
        self.decisions = 0
        self.task = None

    def prompt(self, text: str) -> str:
        return self.tok.apply_chat_template([{"role": "user", "content": text}], tokenize=False,
                                            add_generation_prompt=True, enable_thinking=False)

    async def ask(self, text: str, n: int) -> int:
        if self.queue is None:
            self.queue = asyncio.Queue()
            self.task = asyncio.get_running_loop().create_task(self._loop())
        fut = asyncio.get_running_loop().create_future()
        await self.queue.put((text, n, fut))
        return await fut

    async def _loop(self):
        import torch
        while True:
            items = [await self.queue.get()]
            await asyncio.sleep(0.005)
            while not self.queue.empty() and len(items) < self.max_batch:
                items.append(self.queue.get_nowait())
            batch = self.tok([self.prompt(t) for t, _, _ in items], return_tensors="pt", padding=True).to(self.device)
            with torch.inference_mode():
                logits = self.model(**batch, logits_to_keep=1).logits[:, -1, :]
            digits = logits[:, self.digit_ids].float()
            for row, (_, n, fut) in zip(digits, items):
                if not fut.done():
                    fut.set_result(int(row[:n].argmax()))
            self.decisions += len(items)


async def play(player, opponent, n: int) -> dict:
    t = time.time()
    await player.battle_against(opponent, n_battles=n)
    won = player.n_won_battles
    tied = player.n_tied_battles
    return {"won": won, "tied": tied, "lost": player.n_finished_battles - won - tied,
            "battles": player.n_finished_battles, "win_rate": won / max(1, player.n_finished_battles),
            "seconds": round(time.time() - t, 1)}


def wilson(k: int, n: int, z: float = 1.96) -> list[float]:
    if n == 0:
        return [0.0, 0.0]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return [round(c - h, 4), round(c + h, 4)]


# --------------------------------------------------------------------------- #
# Main job
# --------------------------------------------------------------------------- #

def _train(sample: bool = False, data_battles: int = 3000, eval_battles: int = 200,
           epochs: int = 1, push: bool = False, cpu_smoke: bool = False) -> dict:
    import torch
    from huggingface_hub import HfApi, snapshot_download
    from peft import LoraConfig, PeftModel, get_peft_model
    from poke_env import AccountConfiguration
    from poke_env.player import MaxBasePowerPlayer, RandomPlayer
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if sample:
        data_battles, eval_battles, epochs = 60, 20, 1
    if sample and push:
        raise ValueError("Sample runs must not overwrite the published model")
    started = time.time()
    random.seed(0)
    torch.manual_seed(0)
    device = "cpu" if cpu_smoke else "cuda"
    if not cpu_smoke:
        assert torch.cuda.is_available(), "This entrypoint needs a CUDA GPU"
    token = os.environ.get("HF_TOKEN") or os.environ.get("hf") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    work = Path(os.environ.get("POKEMON_WORK", "/tmp/qwen-pokemon"))
    work.mkdir(parents=True, exist_ok=True)
    out = Path(os.environ.get("COMPUTE_ARTIFACT_DIR", "/tmp/compute-artifacts")) / "qwen-pokemon"
    out.mkdir(parents=True, exist_ok=True)
    (out / ".compute-artifact.json").write_text(json.dumps({
        "name": "qwen-pokemon", "version": 1, "kind": "output",
        "compatibility_key": "qwen3-0.6b-pokemon", "metadata": {"base": BASE, "format": FORMAT}}))
    replays = out / "replays"
    results: dict = {"base": BASE, "revision": REVISION, "format": FORMAT, "sample": sample,
                     "showdown_commit": SHOWDOWN_COMMIT, "node": NODE_VERSION,
                     "data_battles": data_battles, "eval_battles": eval_battles, "epochs": epochs,
                     "gpu": torch.cuda.get_device_name() if not cpu_smoke else "cpu", "eval": {}}

    def save():
        results["elapsed_seconds"] = round(time.time() - started, 1)
        (out / "results.json").write_text(json.dumps(results, indent=2))

    # 1. Showdown -------------------------------------------------------------
    server, timings = start_showdown(work)
    results["setup"] = {k: round(v, 1) for k, v in timings.items()}
    results["setup"]["total_seconds"] = round(sum(timings.values()), 1)
    print("SETUP", json.dumps(results["setup"]), flush=True)
    save()

    Expert, QwenPlayer = make_players()
    counter = iter(range(10 ** 6))

    def acct(prefix):
        return AccountConfiguration(f"{prefix}{next(counter)}", None)

    def kw(prefix, **extra):
        return dict(account_configuration=acct(prefix), battle_format=FORMAT,
                    max_concurrent_battles=32, **extra)

    async def evaluate(name, player):
        opponent = MaxBasePowerPlayer(**kw("maxbp"))
        r = await play(player, opponent, eval_battles)
        r["win_rate_95ci"] = wilson(r["won"], r["battles"])
        results["eval"][name] = r
        print("EVAL", name, json.dumps(r), flush=True)
        save()

    # 2. Expert data and non-LLM baselines --------------------------------------
    async def collect():
        rows: list[dict] = []
        t = time.time()
        # 60% vs the eval opponent's style, 20% vs random, 20% expert self-play.
        plan = [(MaxBasePowerPlayer, 0.6), (RandomPlayer, 0.2), (Expert, 0.2)]
        for cls, share in plan:
            n = max(1, int(data_battles * share))
            expert = Expert(**kw("teach", record=rows))
            opponent = cls(**kw("foe"))
            await expert.battle_against(opponent, n_battles=n)
        results["data"] = {"decisions": len(rows), "seconds": round(time.time() - t, 1)}
        await evaluate("random", RandomPlayer(**kw("rand")))
        await evaluate("expert_no_tera", Expert(**kw("expert")))
        return rows

    rows = asyncio.run(collect())
    random.Random(0).shuffle(rows)
    (out / "expert_decisions.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    answers = [int(r["answer"]) for r in rows]
    results["data"]["answer_histogram"] = {str(d): answers.count(d) for d in range(1, 10)}
    results["data"]["switch_share"] = round(sum(r["switch"] for r in rows) / max(1, len(rows)), 4)
    print("DATA", json.dumps(results["data"]), flush=True)
    save()

    # 3. Model -----------------------------------------------------------------
    t = time.time()
    base_path = snapshot_download(BASE, revision=REVISION, token=token,
                                  allow_patterns=["*.json", "*.safetensors", "*.txt", "LICENSE*"])
    tok = AutoTokenizer.from_pretrained(base_path, padding_side="left")
    dtype = torch.float32 if cpu_smoke else torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(base_path, torch_dtype=dtype, attn_implementation="sdpa").to(device)
    results["model_download_seconds"] = round(time.time() - t, 1)

    def eval_qwen(name, m):
        m.eval()
        batcher = Batcher(m, tok, device)

        async def go():
            await evaluate(name, QwenPlayer(**kw("qwen", batcher=batcher,
                                                   save_replays=str(replays / name))))
        t = time.time()
        asyncio.run(go())
        results["eval"][name]["decisions"] = batcher.decisions
        results["eval"][name]["decisions_per_second"] = round(batcher.decisions / (time.time() - t), 1)
        save()

    # held-out decision accuracy (does the model agree with the expert?)
    holdout = rows[: min(1000, len(rows) // 10)]
    train_rows = rows[len(holdout):]

    def agreement(m, name):
        m.eval()
        b = Batcher(m, tok, device)
        hits = 0
        with torch.inference_mode():
            for s in range(0, len(holdout), 64):
                chunk = holdout[s:s + 64]
                enc = tok([b.prompt(r["prompt"]) for r in chunk], return_tensors="pt", padding=True).to(device)
                logits = m(**enc, logits_to_keep=1).logits[:, -1, b.digit_ids].float()
                for row, r in zip(logits, chunk):
                    hits += int(row[:r["n_options"]].argmax()) + 1 == int(r["answer"])
        results.setdefault("expert_agreement", {})[name] = round(hits / max(1, len(holdout)), 4)
        print("AGREE", name, results["expert_agreement"][name], flush=True)
        save()

    agreement(model, "untuned")
    eval_qwen("qwen_untuned", model)

    # 4. LoRA SFT on the option digit -----------------------------------------
    model = get_peft_model(model, LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.0, task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    model.config.use_cache = False
    records = []
    prompter = Batcher(model, tok, device)
    for r in train_rows:
        ids = tok.encode(prompter.prompt(r["prompt"]), add_special_tokens=False)
        records.append((ids, tok.convert_tokens_to_ids(r["answer"])))
    results["training"] = {"examples": len(records), "holdout": len(holdout),
                           "max_prompt_tokens": max(len(x) for x, _ in records),
                           "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad)}
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=2e-4, weight_decay=0.0)
    batch_size = 4 if cpu_smoke else 32
    steps_total = epochs * ((len(records) + batch_size - 1) // batch_size)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 50) * max(0.0, 1 - s / steps_total))
    losses, step = [], 0
    t = time.time()
    pad = tok.pad_token_id
    for epoch in range(epochs):
        random.Random(epoch).shuffle(records)
        model.train()
        for s in range(0, len(records), batch_size):
            batch = records[s:s + batch_size]
            L = max(len(x) for x, _ in batch)
            # left pad so the answer position is always the last column
            ids = torch.tensor([[pad] * (L - len(x)) + x for x, _ in batch], device=device)
            mask = torch.tensor([[0] * (L - len(x)) + [1] * len(x) for x, _ in batch], device=device)
            target = torch.tensor([y for _, y in batch], device=device)
            logits = model(input_ids=ids, attention_mask=mask, logits_to_keep=1).logits[:, -1, :].float()
            loss = torch.nn.functional.cross_entropy(logits, target)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            losses.append(round(float(loss.detach()), 4))
            step += 1
            if step % 50 == 0 or step == 1:
                print(f"TRAIN epoch {epoch + 1} step {step}/{steps_total} loss {sum(losses[-50:]) / len(losses[-50:]):.4f}",
                      flush=True)
            if cpu_smoke and step >= 3:
                break
    results["training"].update({"steps": step, "seconds": round(time.time() - t, 1), "losses": losses})
    save()
    model.config.use_cache = True
    adapter = out / "adapter"
    model.save_pretrained(adapter)
    tok.save_pretrained(adapter)
    agreement(model, "trained")
    eval_qwen("qwen_sft", model)

    # reload check: the saved adapter reproduces the trained model's choices
    del model
    reloaded = PeftModel.from_pretrained(
        AutoModelForCausalLM.from_pretrained(base_path, torch_dtype=dtype, attn_implementation="sdpa").to(device),
        adapter)
    before = results["expert_agreement"]["trained"]
    agreement(reloaded, "reloaded")
    results["reload_ok"] = results["expert_agreement"]["reloaded"] == before

    server.terminate()
    shutil.copy2(Path(__file__), out / "train.py")
    results["hub_repo"] = None
    if push:
        assert token, "No HF token; use train_and_push with the hf secret"
        card = f"""---
license: apache-2.0
base_model: {BASE}
library_name: peft
tags:
- pokemon
- pokemon-showdown
- poke-env
---
# Qwen3-0.6B plays Pokémon Showdown ({FORMAT})

A LoRA adapter that turns Qwen3-0.6B into a {FORMAT} player. The battle state is
text with numbered options; the model answers one digit and the player takes
the highest-probability legal digit. Trained by imitation on
`SimpleHeuristicsPlayer` decisions (poke-env), tera off.

Win rate against poke-env's `MaxBasePowerPlayer`, {eval_battles} battles each:

```json
{json.dumps({k: {x: v[x] for x in ("won", "battles", "win_rate", "win_rate_95ci")} for k, v in results["eval"].items()}, indent=2)}
```

The prompt builder, player, and training loop are in `train.py`. Guide:
https://letsusecompute.com/posts/qwen-pokemon
"""
        (adapter / "README.md").write_text(card)
        shutil.copy2(out / "train.py", adapter / "train.py")
        save()
        shutil.copy2(out / "results.json", adapter / "results.json")
        api = HfApi(token=token)
        api.create_repo(HUB_REPO, exist_ok=True)
        api.upload_folder(repo_id=HUB_REPO, folder_path=str(adapter),
                          commit_message="Qwen3-0.6B Showdown adapter, SFT on heuristic decisions")
        results["hub_repo"] = HUB_REPO
    save()
    summary = {k: v for k, v in results.items() if k not in ("training",)}
    summary["training"] = {k: v for k, v in results["training"].items() if k != "losses"}
    print("RESULT", json.dumps(summary), flush=True)
    return summary


@app.function(gpu="H100-SXM", image=image, timeout=7200)
def train(sample: bool = False, data_battles: int = 3000, eval_battles: int = 200, epochs: int = 1) -> dict:
    return _train(sample=sample, data_battles=data_battles, eval_battles=eval_battles, epochs=epochs)


@app.function(gpu="H100-SXM", image=image, timeout=7200, secrets=[hf_secret])
def train_and_push(sample: bool = False, data_battles: int = 3000, eval_battles: int = 200, epochs: int = 1) -> dict:
    return _train(sample=sample, data_battles=data_battles, eval_battles=eval_battles, epochs=epochs, push=True)


if __name__ == "__main__":
    # Local smoke test on CPU: python train.py
    print(json.dumps(_train(cpu_smoke=True, data_battles=10, eval_battles=4, epochs=1), indent=2)[:4000])
