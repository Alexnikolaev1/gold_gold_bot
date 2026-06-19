#!/usr/bin/env python3
"""Run system preflight checks."""

from aurum.preflight import run_preflight

if __name__ == "__main__":
    report = run_preflight()
    for c in report.checks:
        status = "OK" if c.ok else "FAIL"
        print(f"[{status}] {c.name}: {c.detail}")
    raise SystemExit(0 if report.all_ok else 1)
