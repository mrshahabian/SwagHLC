"""CLI to browse the RRD dataset — pick subjects/days/trials to stream.

    PYTHONPATH=src python3 -m swag_hlc.dataset_info                 # list subjects
    PYTHONPATH=src python3 -m swag_hlc.dataset_info --subject MP201 # days + trials
"""

from __future__ import annotations

import argparse

from swag_hlc.activities import DEFAULT_ACTIVE_CODES, DEFAULT_ACTIVITY_NAMES, normalize_code
from swag_hlc.dummy_stream import rrd_index


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Browse the RRD dataset")
    ap.add_argument("--root", default=rrd_index.default_root(),
                    required=rrd_index.default_root() is None,
                    help="Path to the RRD dataset root (or set RRD_ROOT env var)")
    ap.add_argument("--subject", default=None)
    ap.add_argument("--include-mvc", action="store_true")
    ap.add_argument("--include-nolabel", action="store_true")
    ap.add_argument("--activities", action="store_true", help="show activity codes + counts")
    args = ap.parse_args(argv)

    if args.subject is None:
        subs = rrd_index.list_subjects(args.root)
        print(f"{len(subs)} subjects in {args.root}:")
        print("  " + ", ".join(subs))
        print("\nRun with --subject <ID> to list its days, trials and activities.")
        return

    for day in rrd_index.list_days(args.root, args.subject):
        trials = rrd_index.list_trials(
            args.root, args.subject, day,
            include_mvc=args.include_mvc, include_nolabel=args.include_nolabel,
        )
        print(f"{args.subject} / {day}: {len(trials)} trials")
        print("  " + ", ".join(trials))

    counts = rrd_index.activity_counts(
        args.root, args.subject, "all",
        include_mvc=args.include_mvc, include_nolabel=args.include_nolabel,
    )
    print(f"\n{args.subject} — activities present (code: samples) "
          f"[* = in default model class set]:")
    active = {normalize_code(c) for c in DEFAULT_ACTIVE_CODES}
    for code in sorted(counts):
        mark = "*" if normalize_code(code) in active else " "
        name = DEFAULT_ACTIVITY_NAMES.get(normalize_code(code), f"activity_{code}")
        print(f"  {mark} {code:>6}  {counts[code]:>9}  {name}")
    print("\n(names are placeholders — set real ones via `activity_names:` in YAML)")


if __name__ == "__main__":
    main()
