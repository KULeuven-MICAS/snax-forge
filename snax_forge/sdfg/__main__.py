import argparse
import json

from .build import run
from .loader import kernel_paths, load


def parse_size(tok: str):
    """'4096' -> 4096;  'm=256,n=512' -> {'m': 256, 'n': 512}"""
    if "=" in tok:
        return {k: int(v) for k, v in (p.split("=") for p in tok.split(","))}
    return int(tok)


def main() -> int:
    ap = argparse.ArgumentParser(prog="snax-forge ingest")
    ap.add_argument("kernels", nargs="*", help="kernel names; empty = all")
    ap.add_argument("--list", action="store_true", help="list available kernel names")
    ap.add_argument("--profile", action="store_true", help="run a size sweep instead of ingest")
    ap.add_argument("--sizes", type=parse_size, nargs="+", help="sweep sizes; int or m=256,n=512")
    ap.add_argument("--reps", type=int, default=50, help="repetitions per size (default: 50)")
    ap.add_argument(
        "--instrument", action="store_true", help="also emit per-state DaCe timers (tier 1)"
    )
    args = ap.parse_args()

    if args.list:
        print("\n".join(sorted(kernel_paths())))
        return 0

    names = args.kernels or sorted(kernel_paths())
    failed = []
    for name in names:
        try:
            if args.profile:
                from .profile import DEFAULT_SIZES, sweep

                sweep(
                    load(name),
                    sizes=args.sizes,
                    reps=args.reps,
                    instrument=args.instrument,
                )
            else:
                report = run(load(name))
                print(json.dumps(report, indent=2))

        except Exception as exc:  # noqa: BLE001
            failed.append((name, repr(exc)))
            print(f"FAILED {name}: {exc!r}")

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
