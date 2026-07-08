"""Command-line entry points: explain, audit, run, trace."""
from __future__ import annotations

import argparse
import sys

import pandas as pd

from sim import config, plots, run as run_mod, score, tariffs as tariffs_mod


def _print_param(p: config.Param) -> None:
    flag = " [TBD]" if p.tbd else ""
    print(f"  {p.name}{flag}")
    print(f"      value  : {p.value!r}")
    print(f"      units  : {p.units}")
    print(f"      meaning: {p.meaning}")
    print(f"      effect : {p.effect}")


def cmd_explain(args: argparse.Namespace) -> None:
    params = config.all_params()
    by_group: dict[str, list[config.Param]] = {g: [] for g in config.GROUP_ORDER}
    for prm in params:
        by_group.setdefault(prm.group, []).append(prm)

    print("=" * 78)
    print("PARAMETER GLOSSARY")
    print("=" * 78)
    for group in config.GROUP_ORDER:
        rows = by_group.get(group, [])
        if not rows:
            continue
        print(f"\n### {group} " + "-" * (74 - len(group)))
        for prm in rows:
            _print_param(prm)
    n_tbd = len(config.tbd_params())
    print("\n" + "=" * 78)
    print(f"{n_tbd}/{len(params)} parameters are flagged [TBD] (placeholder guesses). "
          f"Run `python -m sim audit` to list just those.")


def cmd_audit(args: argparse.Namespace) -> None:
    tbd = config.tbd_params()
    print(f"{len(tbd)} parameters are placeholder guesses (tbd=True):\n")
    name_w = max((len(p.name) for p in tbd), default=4)
    value_w = max((len(repr(p.value)) for p in tbd), default=5)
    header = f"{'name'.ljust(name_w)}  {'value'.ljust(value_w)}  meaning"
    print(header)
    print("-" * len(header))
    for prm in tbd:
        print(f"{prm.name.ljust(name_w)}  {repr(prm.value).ljust(value_w)}  {prm.meaning}")


def cmd_run(args: argparse.Namespace) -> None:
    n_tbd = len(config.tbd_params())
    print(f"NOTE: {n_tbd} parameters are placeholder guesses -- run `python -m sim audit` to list them.\n")

    tariff_names = args.tariffs or list(tariffs_mod.CANDIDATES.keys())
    results, population = run_mod.run_sweep(
        tariff_names, scenario_name=args.scenario, seed=args.seed, R=args.R, n_agents=args.n_agents,
        no_hunger=args.no_hunger, no_cost=args.no_cost, no_personas=args.no_personas,
        trace_agent=args.trace,
    )

    board = score.scoreboard(results, population)
    print("SCOREBOARD (sorted by wood_share ascending)")
    print(board.to_string(index=False))
    board.to_csv(f"{args.out_dir}/scoreboard.csv", index=False)

    paths = plots.make_all_plots(results, population, out_dir=args.out_dir)
    print("\nWrote plots:")
    for path in paths:
        print(f"  {path}")

    if args.trace is not None:
        tname = tariff_names[0]
        trace_rows = results[tname].trace_rows
        if not trace_rows:
            print(f"\nNo trace rows captured for agent {args.trace} (check --n-agents > id).")
        else:
            df = pd.DataFrame(trace_rows)
            trace_path = f"{args.out_dir}/trace_agent_{args.trace}.csv"
            df.to_csv(trace_path, index=False)
            print(f"\nWrote decision trace ({tname} tariff, run 0) for agent {args.trace}: {trace_path}")
            eaten = df[df["fired"]]
            cols = ["block", "t_hr", "stage_idx", "choice_name", "duration_blocks", "q", "hunger"]
            cols = [c for c in cols if c in eaten.columns]
            if len(eaten):
                print("\nEaten-meal blocks:")
                print(eaten[cols].to_string(index=False))
            else:
                print(f"\nAgent {args.trace} did not eat anything on this traced day.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m sim", description="Clean-cooking mini-grid tariff simulator")
    sub = parser.add_subparsers(dest="command", required=True)

    p_explain = sub.add_parser("explain", help="Print the full parameter glossary.")
    p_explain.set_defaults(func=cmd_explain)

    p_audit = sub.add_parser("audit", help="List all [TBD] placeholder parameters.")
    p_audit.set_defaults(func=cmd_audit)

    p_run = sub.add_parser("run", help="Run the tariff sweep: simulate, score, plot.")
    p_run.add_argument("--tariffs", nargs="+", choices=list(tariffs_mod.CANDIDATES.keys()), default=None,
                        help="Tariffs to sweep (default: all candidates).")
    p_run.add_argument("--scenario", default="reference", choices=list(config.SCENARIOS.keys()))
    p_run.add_argument("--seed", type=int, default=config.DEFAULT_SEED)
    p_run.add_argument("--R", type=int, default=config.SCORING.R, help="Monte Carlo runs per tariff.")
    p_run.add_argument("--n-agents", type=int, default=config.N_AGENTS)
    p_run.add_argument("--out-dir", default="out")
    p_run.add_argument("--no-cost", action="store_true", help="Ablation: zero every agent's price sensitivity.")
    p_run.add_argument("--no-hunger", action="store_true", help="Ablation: zero hunger's effect on firing/choice.")
    p_run.add_argument("--no-personas", action="store_true", help="Ablation: force everyone to base household.")
    p_run.add_argument("--trace", type=int, default=None, metavar="AGENT_ID",
                        help="Log every block for this agent (first tariff, run 0) to a CSV.")
    p_run.set_defaults(func=cmd_run)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main(sys.argv[1:])
