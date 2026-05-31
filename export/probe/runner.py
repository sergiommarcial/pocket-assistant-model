from __future__ import annotations
import subprocess
import sys
from pathlib import Path

from .config import Config
from .suite import Probe, PROBES

SYSTEM = Path("data/system_prompt.txt").read_text().strip()

CHATML = (
    "<|im_start|>system\n{system}<|im_end|>\n"
    "<|im_start|>user\n{user}<|im_end|>\n"
    "<|im_start|>assistant\n"
)


def build_prompt(user: str) -> str:
    return CHATML.format(system=SYSTEM, user=user)


def generate(
    model_path: str, prompt: str, max_tokens: int, verbose: bool = False
) -> str:
    if verbose:
        print(f"  [generate] model={model_path}", file=sys.stderr)
        print(f"  [generate] prompt_tail={prompt[-120:].strip()!r}", file=sys.stderr)

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "mlx_lm.generate",
            "--model",
            model_path,
            "--prompt",
            prompt,
            "--max-tokens",
            str(max_tokens),
            "--temp",
            "0.0",
        ],
        capture_output=True,
        text=True,
    )

    if verbose and result.stderr.strip():
        print(f"  [generate] stderr={result.stderr.strip()!r}", file=sys.stderr)

    output = result.stdout
    if "==========" in output:
        parts = output.split("==========")
        return parts[1].strip() if len(parts) >= 3 else output.strip()
    return output.strip()


def run_probes(config: Config) -> None:
    print(f"Probing model: {config.model_path}", file=sys.stderr)

    passed = 0
    failed_blocking: list[tuple[Probe, str, list[str]]] = []
    failed_advisory: list[tuple[Probe, str, list[str]]] = []
    domain_counts: dict[str, list[int]] = {}  # domain -> [passed, total]

    by_category: dict[str, list[Probe]] = {}
    for p in PROBES:
        by_category.setdefault(p.category, []).append(p)

    for category, probes in by_category.items():
        domains = sorted({p.domain for p in probes})
        for domain in domains:
            domain_probes = [p for p in probes if p.domain == domain]
            print(f"\n{'='*60}")
            print(f"  {category} · {domain}  ({len(domain_probes)} probes)")
            print(f"{'='*60}")

            for probe in domain_probes:
                prompt = build_prompt(probe.user)
                response = generate(
                    config.model_path, prompt, config.max_tokens, verbose=config.verbose
                )
                response_lower = response.lower()

                check_failures = []
                for kw in probe.must_contain:
                    if kw.lower() not in response_lower:
                        check_failures.append(f"missing '{kw}'")
                for kw in probe.must_not_contain:
                    if kw.lower() in response_lower:
                        check_failures.append(f"contains '{kw}'")

                dc = domain_counts.setdefault(probe.domain, [0, 0])
                dc[1] += 1

                status = "PASS" if not check_failures else "FAIL"
                suffix = "" if probe.blocking else " [advisory]"
                if status == "PASS":
                    passed += 1
                    dc[0] += 1
                elif probe.blocking:
                    failed_blocking.append((probe, response, check_failures))
                else:
                    failed_advisory.append((probe, response, check_failures))

                print(f"\n[{status}] {probe.label}{suffix}")
                print(f"  > {response[:220].strip()}")
                if check_failures:
                    for f in check_failures:
                        print(f"  ! {f}")

    total = passed + len(failed_blocking) + len(failed_advisory)
    print(f"\n{'='*60}")
    print(f"  Results: {passed}/{total} passed")
    print(f"{'='*60}\n")

    # ── Approval gates ────────────────────────────────────────────────── #
    threshold_violations: list[str] = []
    print(f"{'='*60}")
    print(f"  APPROVAL GATES")
    print(f"{'='*60}\n")
    for domain in sorted(domain_counts):
        domain_passed, domain_total = domain_counts[domain]
        rate = domain_passed / domain_total if domain_total > 0 else 1.0
        minimum = config.thresholds.get(domain, 1.0)
        gate = "PASS" if rate >= minimum else "FAIL"
        cmp = "≥" if rate >= minimum else "<"
        print(
            f"  [{gate}] {domain}: {domain_passed}/{domain_total}"
            f" ({rate:.1%} {cmp} {minimum:.0%})"
        )
        if gate == "FAIL":
            threshold_violations.append(domain)

    if failed_advisory:
        print(f"\n  Advisory (known failures — tracked, not blocking):")
        for probe, _, _ in failed_advisory:
            print(f"    · {probe.label}")

    if failed_blocking or threshold_violations:
        print(f"\n{'='*60}")
        if failed_blocking:
            print(f"  BLOCKING FAILURES ({len(failed_blocking)})")
            print(f"{'='*60}")
            for probe, response, check_failures in failed_blocking:
                print(
                    f"\n[FAIL] {probe.label}  [{probe.category} · {probe.domain}]"
                )
                print(f"  > {response[:220].strip()}")
                for f in check_failures:
                    print(f"  ! {f}")
        if threshold_violations:
            print(f"\n  Threshold violations: {', '.join(threshold_violations)}")
        print()
        sys.exit(1)
