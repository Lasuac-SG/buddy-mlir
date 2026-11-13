#!/usr/bin/env python3
# ===- benchmark.py - EmbeddingGemma CLI baseline measurements -------------===//
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# ===----------------------------------------------------------------------===//

import argparse
import contextlib
import json
import math
import os
import platform
import re
import shutil
import signal
import statistics
import subprocess
import threading
import time
from pathlib import Path

import numpy as np
from validate_accuracy import compare_outputs


def cpu_model():
    for line in Path("/proc/cpuinfo").read_text().splitlines():
        key, _, value = line.partition(":")
        if (
            key.strip() in ("model name", "uarch", "Processor")
            and value.strip()
        ):
            return value.strip()
    return platform.machine()


def run_once(args, index, text, directory, env):
    prefix = directory / f"sample_{index}"
    command = [
        args.time_command,
        "-q",
        "-f",
        "%M",
        "-o",
        str(prefix) + ".rss_kib",
        str(args.buddy_cli),
        "--model",
        str(args.model),
    ]
    if text:
        command.extend(["--prompt", text])
    else:
        # A blank line avoids buddy-cli's interactive empty-prompt path.
        command.extend(
            ["--prompt-file", str(args.output_dir / "empty_prompt.txt")]
        )

    with (
        open(str(prefix) + ".stdout", "w") as stdout,
        open(str(prefix) + ".stderr", "w") as stderr,
    ):
        start = time.perf_counter_ns()
        with subprocess.Popen(
            command,
            stdout=stdout,
            stderr=stderr,
            env=env,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        ) as process:
            timed_out = threading.Event()

            def kill_group():
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGKILL)

            def expire():
                timed_out.set()
                kill_group()

            watchdog = threading.Timer(args.timeout, expire)
            watchdog.daemon = True
            watchdog.start()
            try:
                # Blocking wait avoids the polling delay of wait(timeout=...).
                returncode = process.wait()
                elapsed_ms = (time.perf_counter_ns() - start) / 1e6
            finally:
                watchdog.cancel()
                watchdog.join()
                if process.poll() is None:
                    kill_group()
                    process.wait()
            if timed_out.is_set():
                raise subprocess.TimeoutExpired(command, args.timeout)
    if returncode:
        raise RuntimeError(
            f"Runner exited with status {returncode}; see {prefix}.stderr"
        )

    log = Path(str(prefix) + ".stderr").read_text()
    timing = re.findall(r"^\s*time\s*:\s*([0-9.eE+-]+)s\s*$", log, re.M)
    sequence = re.findall(r"^\s*seq_len\s*:\s*(\d+)\s*$", log, re.M)
    if len(timing) != 1 or sequence != [str(args.sequence_length)]:
        raise ValueError(
            f"Missing timing or unexpected sequence length in {prefix}.stderr"
        )
    forward_ms = float(timing[0]) * 1000
    peak_rss_kib = int(Path(str(prefix) + ".rss_kib").read_text())
    if not math.isfinite(forward_ms) or forward_ms <= 0 or peak_rss_kib <= 0:
        raise ValueError(f"Invalid timing or peak RSS for sample {index}")

    embedding = np.asarray(
        json.loads(Path(str(prefix) + ".stdout").read_text()), dtype=np.float32
    )
    if (
        embedding.shape != (args.embedding_dim,)
        or not np.isfinite(embedding).all()
    ):
        raise ValueError(f"Invalid embedding for sample {index}")
    np.save(directory / f"buddy_embedding_{index}.npy", embedding[None, :])
    return {
        "e2e_wall_ms": elapsed_ms,
        "forward_first_ms": forward_ms,
        "peak_rss_mib": peak_rss_kib / 1024,
    }


def summarize(values):
    return {
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
        "stddev": statistics.pstdev(values),
    }


def main():
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Measure the EmbeddingGemma CLI baseline (Linux/GNU time)."
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="New directory for results and payload cache",
    )
    parser.add_argument(
        "--buddy-cli",
        type=Path,
        default=here.parents[1] / "build/bin/buddy-cli",
    )
    parser.add_argument("--spec", type=Path, default=here / "specs/f32.json")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument(
        "--warmup",
        type=int,
        default=1,
        help="Unmeasured cache-preparation rounds (at least 1)",
    )
    parser.add_argument("--repeat", type=int, default=10)
    parser.add_argument(
        "--timeout",
        type=float,
        default=600,
        help="Timeout per Runner process, in seconds",
    )
    parser.add_argument("--tolerance", type=float, default=1e-5)
    args = parser.parse_args()
    if min(args.threads, args.warmup, args.repeat) < 1:
        parser.error("threads, warmup and repeat must be positive")
    if not all(
        math.isfinite(x) and x > 0 for x in (args.timeout, args.tolerance)
    ):
        parser.error("timeout and tolerance must be finite and positive")
    args.time_command = shutil.which("time")
    if platform.system() != "Linux" or not args.time_command:
        parser.error("Linux and GNU time are required")
    for path in (args.model, args.buddy_cli, args.spec):
        if not path.is_file():
            parser.error(f"File not found: {path}")

    with args.spec.open() as stream:
        spec = json.load(stream)
    args.sequence_length = int(spec["max_seq_len"])
    args.embedding_dim = int(spec["hidden_size"])
    with (args.reference_dir / "reference_manifest.json").open() as stream:
        references = json.load(stream)
    cases = []
    for key, reference in references.items():
        if not key.startswith("text_"):
            continue
        if not re.fullmatch(r"text_[0-9]+", key) or not isinstance(
            reference["text"], str
        ):
            parser.error(f"Invalid reference sample: {key}")
        index = key.removeprefix("text_")
        if int(reference["embedding_dim"]) != args.embedding_dim:
            parser.error(f"Unexpected reference dimension for {key}")
        if not (args.reference_dir / f"ref_embedding_{index}.npy").is_file():
            parser.error(f"Missing reference embedding for {key}")
        cases.append((index, reference["text"]))
    if not cases:
        parser.error("No reference samples found")
    cases.sort(key=lambda case: int(case[0]))
    args.output_dir = args.output_dir.resolve()
    if args.output_dir.exists():
        parser.error("Output directory already exists; choose a new directory")
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "empty_prompt.txt").write_text("\n")
    env = dict(
        os.environ,
        OMP_NUM_THREADS=str(args.threads),
        OMP_DYNAMIC="FALSE",
        LC_ALL="C",
        BUDDY_RAX_PAYLOAD_DIR=str(args.output_dir / "payload"),
    )
    report = {
        "model": str(args.model),
        "model_size_mib": args.model.stat().st_size / 1024**2,
        "cpu": cpu_model(),
        "threads": args.threads,
        "dtype": spec["variant"],
        "batch_size": 1,
        "sequence_length": args.sequence_length,
        "warmup": args.warmup,
        "repeat": args.repeat,
        "tolerance": args.tolerance,
        "validation_passed": False,
        "samples": [
            {"id": index, "text": text, "runs": []} for index, text in cases
        ],
    }
    try:
        for phase, count in (("warmup", args.warmup), ("runs", args.repeat)):
            for iteration in range(count):
                directory = args.output_dir / phase / f"{iteration:03d}"
                directory.mkdir(parents=True)
                for sample in report["samples"]:
                    print(
                        f"[{phase} {iteration + 1}/{count}] sample {sample['id']}",
                        flush=True,
                    )
                    measured = run_once(
                        args, sample["id"], sample["text"], directory, env
                    )
                    if phase == "runs":
                        sample["runs"].append(measured)
        # Validate every measured round after all timing has finished.
        passed = True
        for iteration in range(args.repeat):
            directory = args.output_dir / "runs" / f"{iteration:03d}"
            with (
                (directory / "validation.log").open("w") as stream,
                contextlib.redirect_stdout(stream),
            ):
                round_passed = compare_outputs(
                    str(args.reference_dir), str(directory), args.tolerance
                )
            passed = round_passed and passed
        report["validation_passed"] = passed
        for sample in report["samples"]:
            for name in ("e2e_wall_ms", "forward_first_ms"):
                sample[name] = summarize([run[name] for run in sample["runs"]])
            sample["peak_rss_mib_max"] = max(
                run["peak_rss_mib"] for run in sample["runs"]
            )
        report["peak_rss_mib_max"] = max(
            sample["peak_rss_mib_max"] for sample in report["samples"]
        )
        if not passed:
            raise ValueError(
                "Numerical validation failed; see runs/*/validation.log"
            )
    except (
        OSError,
        ValueError,
        RuntimeError,
        subprocess.SubprocessError,
        KeyboardInterrupt,
    ) as error:
        report["error"] = str(error) or type(error).__name__
        report["validation_passed"] = False
    (args.output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )
    print(f"Report: {args.output_dir / 'report.json'}")
    if "error" in report:
        print(f"ERROR: {report['error']}")
        return 1
    for sample in report["samples"]:
        print(
            f"sample {sample['id']}: e2e median={sample['e2e_wall_ms']['median']:.2f} ms, forward median={sample['forward_first_ms']['median']:.2f} ms, peak RSS={sample['peak_rss_mib_max']:.2f} MiB"
        )
    print("Validation: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
