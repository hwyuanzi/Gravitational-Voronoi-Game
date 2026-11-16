"""Tournaments from the command line (the dashboard uses the same code).

  python3 start.py tournament                                  # every pair of bots
  python3 start.py tournament --melee                          # one melee with every bot
  python3 start.py tournament --config params/s10_w100.txt     # one parameter file throughout
  python3 start.py tournament --bots "Greedy (Python)" "Random (Python)"

A pairing is one match: two games with the same parameters, each bot moving first once, and
a bot's score is its territory over both games added together. A melee is the same with more
bots: one game per bot, the seating rotating so every bot sits in every seat once.
"""
import argparse
import itertools
import json
import sys
import time
from pathlib import Path

from .engine import load_config
from .runner import ROOT, build, load_bots, play_match

PARAM_FOLDERS = ('params',)


def list_params(folders=PARAM_FOLDERS):
    """Every .txt parameter file in the given folders, with its stones and weight."""
    if isinstance(folders, (str, Path)):
        folders = [folders]
    out = []
    paths = [p for f in folders for p in sorted((ROOT / f).glob('*.txt'))]
    for path in paths:
        rel = path.resolve().relative_to(ROOT).as_posix()
        entry = {'file': rel, 'name': rel[:-4]}
        try:
            cfg = load_config(path)
            entry.update(cfg.as_dict())
            if cfg.warning:
                entry['warning'] = cfg.warning
        except (OSError, ValueError) as exc:
            entry['error'] = str(exc)
        out.append(entry)
    return out


def pairing_result(state):
    """The fields the results table needs from one finished match."""
    s = state['summary']
    return {'totals': s['totals'], 'winner': s['winner'], 'games': s['games'],
            'params': state['params'], 'one_game': state.get('one_game', False)}


def build_all(bots):
    """Compile bots that need it, before any clock runs. Returns {name: error or None}."""
    problems = {}
    for b in bots:
        try:
            build(b)
            problems[b['name']] = None
        except Exception as exc:
            problems[b['name']] = str(exc)
    return problems


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--bots', nargs='+', help='bot names from bots.json (default: all)')
    ap.add_argument('--melee', action='store_true', help='one match with all the bots at once')
    ap.add_argument('--registry', default='bots.json')
    ap.add_argument('--config', help='use this one parameter file for every match')
    ap.add_argument('--clock', type=float, default=120, help='seconds per bot per game')
    ap.add_argument('--one-game', action='store_true',
                    help='every match is one game, the first bot named moving first')
    ap.add_argument('--output', default='results/tournament')
    args = ap.parse_args()

    roster = {b['name']: b for b in load_bots(args.registry)}
    names = args.bots or list(roster)
    missing = [n for n in names if n not in roster]
    if len(names) < 2 or missing:
        raise ValueError(f'Pick at least two bots from: {", ".join(roster)}')
    files = [args.config] if args.config else [p['file'] for p in list_params() if 'error' not in p]
    if not files:
        raise ValueError('No valid parameter files in params/')
    out = ROOT / args.output / time.strftime('%Y%m%d-%H%M%S')

    for name, err in build_all([roster[n] for n in names]).items():
        if err:
            print(f'! {name}: {err}')
    groups = [tuple(names)] if args.melee else list(itertools.combinations(names, 2))
    print(f'{len(names)} bots, {len(groups)} {"melee" if args.melee else "pairings"}\n')
    results = []
    table = {n: 0 for n in names}
    for i, group in enumerate(groups):
        config = files[i % len(files)]
        print(f'[{i + 1}/{len(groups)}] {" vs ".join(group)} ({Path(config).name})', flush=True)
        state = play_match([roster[n] for n in group], config, args.clock, out / f'{i + 1:02d}',
                           echo=True, one_game=args.one_game)
        entry = {'bots': list(group), 'config': config}
        entry.update(pairing_result(state))
        tot = entry['totals']
        print('    ' + ', '.join(f'{n} {tot.get(n, "?")}' for n in group) +
              f"  ->  {entry['winner'] or 'tie'}\n", flush=True)
        for n in group:
            table[n] += tot.get(n, 0)
        results.append(entry)
        (out / 'results.json').write_text(json.dumps(results, indent=2), encoding='utf-8')

    print('Territory over the whole tournament:')
    for name, pts in sorted(table.items(), key=lambda kv: -kv[1]):
        print(f'  {pts:9}  {name}')
    print(f'\nResults, replays and logs: {out.relative_to(ROOT)}/')


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\nStopped. Finished matches are saved.')
    except (OSError, ValueError) as exc:
        sys.exit(f'Error: {exc}')
