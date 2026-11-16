"""Play one match between two or more bots from bots.json and open the replay.

  python3 start.py match "Greedy (Python)" "Random (Python)"
  python3 start.py match "A" "B" "C" --config params/s10_w100.txt      # a three-bot melee

One game per bot, the seating rotating so each bot moves first once. A bot's score is its
territory over all the games added together. With --one-game they play a single game instead,
the first bot named moving first. Results go to results/match/ (results.json,
replay.html and log.txt). For many bots at once, use the dashboard: python3 start.py
"""
import argparse
import sys
import webbrowser

from .engine import COLORS
from .runner import ROOT, build, load_bots, play_match


def describe(game):
    s = game.get('scores') or []
    colors = game.get('colors') or COLORS
    parts = [f'{n} ({colors[i]}) {s[i] if i < len(s) else "?"}' for i, n in enumerate(game['seats'])]
    extra = []
    for i, n in enumerate(game['seats']):
        if (game.get('skipped') or [0] * 99)[i]:
            extra.append(f"{n}: {game['skipped'][i]} turn(s) skipped")
        if (game.get('errors') or [None] * 99)[i]:
            extra.append(f"{n}: forfeit ({game['errors'][i]})")
    return f"Game {game['game']}: " + ' - '.join(parts) + (f"  [{'; '.join(extra)}]" if extra else '')


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('bots', nargs='+', metavar='bot', help='two or more bot names')
    ap.add_argument('--config', default='params/s10_w100.txt')
    ap.add_argument('--clock', type=float, default=120, help='seconds per bot per game')
    ap.add_argument('--registry', default='bots.json')
    ap.add_argument('--output', default='results/match')
    ap.add_argument('--no-browser', action='store_true')
    ap.add_argument('--one-game', action='store_true',
                    help='play one game only, the first bot named moving first')
    args = ap.parse_args()
    if len(args.bots) < 2:
        sys.exit('Name at least two bots.')
    roster = {b['name']: b for b in load_bots(args.registry)}
    for name in args.bots:
        if name not in roster:
            sys.exit(f'Unknown bot {name!r}. Known bots: {", ".join(roster)}')
        build(roster[name])
    st = play_match([roster[n] for n in args.bots], args.config, clock=args.clock,
                    out_dir=ROOT / args.output, echo=True, one_game=args.one_game)
    for g in st['summary']['games']:
        print(describe(g))
    totals = st['summary']['totals']
    print('Totals: ' + ', '.join(f'{n} {p}' for n, p in totals.items()))
    w = st['summary']['winner']
    print('Winner: ' + (w if w else 'tie'))
    replay = ROOT / args.output / 'replay.html'
    print(f'Replay: {replay}\nLog:    {ROOT / args.output / "log.txt"}')
    if not args.no_browser:
        webbrowser.open(replay.as_uri())


if __name__ == '__main__':
    main()
