import argparse
import json
import shutil
import subprocess
import time
from pathlib import Path

from jinja2 import StrictUndefined, Template
from minisweagent.run.benchmarks.swebench import process_instance
from minisweagent.run.benchmarks.utils.batch_progress import RunBatchProgressManager
from minisweagent.utils.log import add_file_handler

ROOT = Path(__file__).resolve().parents[2]
CORPUS = ROOT / "workspace" / "external" / "rollout_dialogues" / "origin"
REFERENCE = "swebench-origin-claude-sonnet4.6"
STATEMENT_START = "<pr_description>\nConsider the following PR description:\n"
STATEMENT_END = "\n</pr_description>\n\n<instructions>"


def corpus_instances(instance_template: str) -> list[dict]:
    folders = [folder for folder in sorted(CORPUS.iterdir()) if folder.name.startswith("swebench-origin-")]
    paths = {folder.name: {p.name.removesuffix(".traj.json"): p for p in folder.rglob("*.traj.json")} for folder in folders}
    shared = sorted(set.intersection(*(set(ids) for ids in paths.values())))
    instances = []
    for instance_id in shared:
        document = json.loads(paths[REFERENCE][instance_id].read_text(encoding="utf-8"))
        first_user = next(m["content"] for m in document["messages"] if m["role"] == "user")
        statement = first_user[len(STATEMENT_START) : first_user.index(STATEMENT_END)]
        if Template(instance_template, undefined=StrictUndefined).render(task=statement) != first_user:
            raise ValueError(f"{instance_id}: recovered statement does not reproduce the reference prompt")
        instances.append({
            "instance_id": instance_id,
            "problem_statement": statement,
            "image_name": document["info"]["config"]["environment"]["image"],
        })
    return instances


def docker(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["docker", *args], capture_output=True, text=True, encoding="utf-8", errors="replace")


def drop_vm_page_cache() -> None:


    if shutil.which("wsl") is None:
        return
    subprocess.run(["wsl", "-d", "docker-desktop", "sh", "-c", "sync; echo 3 > /proc/sys/vm/drop_caches"], capture_output=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tasks", default="", help="comma-separated instance ids; default all shared tasks")
    parser.add_argument("--instances", type=Path, help="task list from --write-instances; default read the corpus")
    parser.add_argument("--write-instances", type=Path, help="write the derived task list and exit")
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    if args.instances is not None:
        instances = json.loads(args.instances.read_text(encoding="utf-8"))
    else:
        instances = corpus_instances(config["agent"]["instance_template"])
    if args.write_instances is not None:
        args.write_instances.write_text(json.dumps(instances, indent=2) + chr(10), encoding="utf-8", newline=chr(10))
        print(f"{len(instances)} instances -> {args.write_instances}")
        return
    if args.tasks:
        wanted = args.tasks.split(",")
        instances = [instance for instance in instances if instance["instance_id"] in wanted]
    args.output.mkdir(parents=True, exist_ok=True)
    add_file_handler(args.output / "minisweagent.log")
    progress = RunBatchProgressManager(len(instances), args.output / f"exit_statuses_{time.time()}.yaml")
    ledger = args.output / "tasks.jsonl"

    for instance in instances:
        instance_id, image = instance["instance_id"], instance["image_name"]
        trajectory = args.output / instance_id / f"{instance_id}.traj.json"
        if trajectory.exists():
            continue
        started = time.time()
        for attempt in range(3):

            pull = docker("pull", image)
            if pull.returncode == 0 or attempt == 2:
                break
            time.sleep(60)
        drop_vm_page_cache()
        pulled = time.time()
        record = {"instance_id": instance_id, "image": image, "pull_seconds": round(pulled - started, 1)}
        if pull.returncode == 0:
            try:
                process_instance(instance, args.output, config, progress)
            finally:
                containers = docker("ps", "-aq", "--filter", f"ancestor={image}").stdout.split()
                if containers:
                    docker("rm", "-f", *containers)
                record["rmi_returncode"] = docker("rmi", image).returncode
                drop_vm_page_cache()
            record["run_seconds"] = round(time.time() - pulled, 1)
            if trajectory.exists():
                record["exit_status"] = json.loads(trajectory.read_text(encoding="utf-8"))["info"]["exit_status"]
        else:
            record["pull_error"] = pull.stderr.strip()[-500:]
        with ledger.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
