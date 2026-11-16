"""Check a finished match against the rules, independently of the referee.

  python3 start.py verify results/match/results.json
  python3 start.py verify results/tournament/20261031-151232/01/results.json

This file does NOT use the referee's engine. It re-reads the saved match and checks, from
scratch, every rule in the spec:

  1. one game per bot, with the seats rotating so each bot moves first once
  2. players take turns in seat order, one stone per turn, stones_each turns each,
     so the last turn belongs to the last seat
  3. every placed stone is on the 1000 x 1000 board at whole-number coordinates
  4. every placed stone is at least 66 from every stone placed before it
  5. every weight is a whole number >= 1, and no player spends more than its total weight
  6. every skipped turn really was illegal (the stone the bot sent is re-checked)
  7. a bot that forfeited is asked nothing more that game
  8. no clock goes below zero or above the budget
  9. the final territory of each game, recounted point by point with the plain formula
     pull = sum(weight / distance^2), with no shortcuts
 10. the match totals are the sum over the games, and the winner has the highest total

It prints PASS or FAIL for each rule. Recounting a 1000 x 1000 board in plain Python takes a
few seconds per game.
"""
import argparse
import json
import sys
from pathlib import Path

GRID = 1000
MIN_DIST = 66
TIE = 1e-9


class Report:
    def __init__(self):
        self.failures = {}

    def check(self, rule, ok, detail):
        if not ok:
            self.failures.setdefault(rule, []).append(detail)


def whole(v):
    if isinstance(v, bool):
        return None
    if isinstance(v, int):
        return v
    if isinstance(v, float) and v.is_integer():
        return int(v)
    if isinstance(v, str):
        try:
            return whole(json.loads(v.strip()))
        except ValueError:
            return None
    return None


def recount(stones, k, grid):
    """Who owns every point, the slow obvious way. Returns points owned per player."""
    at = {(x, y): p for p, x, y, w in stones}
    counts = [0] * k
    for y in range(grid):
        for x in range(grid):
            if (x, y) in at:
                counts[at[(x, y)]] += 1
                continue
            pull = [0.0] * k
            for p, sx, sy, w in stones:
                pull[p] += w / ((x - sx) ** 2 + (y - sy) ** 2)
            best = max(pull)
            if best <= 0:
                continue
            leaders = [p for p in range(k) if pull[p] * (1 + TIE) >= best]
            if len(leaders) == 1:
                counts[leaders[0]] += 1
    return counts


def verify(path, skip_recount=False):
    st = json.loads(Path(path).read_text(encoding='utf-8'))
    r = Report()
    names = st['names']
    k = len(names)
    P = st['params']
    S, W = P['stones'], P['weight']
    grid = st.get('grid', GRID)
    min_dist = st.get('min_dist', MIN_DIST)
    budget = st['time_budget']
    games = st['games']
    print(f"Match: {' vs '.join(names)}   {S} stones and {W} weight each, "
          f"{budget:g}s per bot per game, {grid}x{grid} board, min distance {min_dist}")
    if (grid, min_dist) != (GRID, MIN_DIST):
        print(f'  note: this match was not played on the spec board ({GRID}, {MIN_DIST})')

    # 1. rotation (or a single game in the order the bots were named)
    one = st.get('one_game', False)
    r.check(1, len(games) == (1 if one else k), f'{len(games)} games for {k} bots'
                                                 f"{' (one-game match)' if one else ''}")
    for i, g in enumerate(games):
        want = [names[(i + s) % k] for s in range(k)]
        r.check(1, g['seats'] == want, f"game {g['game']}: seats {g['seats']}, expected {want}")

    totals = {n: 0 for n in names}
    for g in games:
        gid = g['game']
        moves = g['moves']
        # 2. turn order and count
        r.check(2, len(moves) == k * S, f'game {gid}: {len(moves)} turns, expected {k * S}')
        for t, m in enumerate(moves):
            r.check(2, m['player'] == t % k, f"game {gid} turn {t + 1}: seat {m['player'] + 1} moved, "
                                             f'expected seat {t % k + 1}')
        if moves:
            r.check(2, moves[-1]['player'] == k - 1, f'game {gid}: the last turn was not the last seat')

        placed, spent, forfeited = [], [0] * k, [False] * k
        for t, m in enumerate(moves):
            p = m['player']
            s = m.get('stone')
            where = f'game {gid} turn {t + 1} ({g["seats"][p]})'
            # 7. nothing after a forfeit
            if forfeited[p]:
                r.check(7, s is None and m.get('forfeit'), f'{where}: played after forfeiting')
            if m.get('forfeit'):
                forfeited[p] = True
                r.check(7, s is None, f'{where}: a forfeited turn placed a stone')
            if s is not None:
                x, y, w = s['x'], s['y'], s['weight']
                # 3. on the board, whole numbers
                r.check(3, all(isinstance(v, int) and not isinstance(v, bool) for v in (x, y, w)),
                        f'{where}: non-integer stone {s}')
                r.check(3, 0 <= x < grid and 0 <= y < grid, f'{where}: ({x}, {y}) is off the board')
                # 4. distance to every earlier stone
                for q, a, b, _ in placed:
                    d2 = (x - a) ** 2 + (y - b) ** 2
                    r.check(4, d2 >= min_dist ** 2,
                            f'{where}: ({x}, {y}) is {d2 ** .5:.1f} from ({a}, {b})')
                # 5. weight
                r.check(5, w >= 1, f'{where}: weight {w}')
                spent[p] += w
                r.check(5, spent[p] <= W, f'{where}: has now spent {spent[p]} of {W}')
                r.check(5, s['player'] == p, f'{where}: stone credited to the wrong player')
                placed.append((p, x, y, w))
            elif m.get('refused'):
                # 6. the skip must have been deserved
                sent = m.get('sent') or {}
                sx, sy, sw = whole(sent.get('x')), whole(sent.get('y')), whole(sent.get('weight'))
                illegal = (sx is None or sy is None or sw is None
                           or not (0 <= sx < grid and 0 <= sy < grid)
                           or sw < 1 or sw > W - spent[p]
                           or any((sx - a) ** 2 + (sy - b) ** 2 < min_dist ** 2 for _, a, b, _ in placed))
                r.check(6, illegal, f'{where}: skipped, but {sent} looks legal')
            # 8. clocks
            for c in m.get('clocks', []):
                r.check(8, 0 <= c <= budget, f'{where}: a clock reads {c}')

        # 9. territory
        final = g['scores']
        if skip_recount:
            print(f'  game {gid}: recount skipped')
        else:
            print(f'  game {gid}: recounting {grid * grid:,} points ...', end='', flush=True)
            mine = recount(placed, k, grid)
            print(' done')
            r.check(9, mine == final, f'game {gid}: referee says {final}, recount says {mine}')
        for seat, n in enumerate(g['seats']):
            totals[n] += final[seat]

    # 10. totals and winner
    summary = st['summary']
    r.check(10, summary['totals'] == totals, f"totals {summary['totals']}, expected {totals}")
    best = max(totals.values())
    leaders = [n for n, v in totals.items() if v == best]
    want = leaders[0] if len(leaders) == 1 else None
    r.check(10, summary['winner'] == want, f"winner {summary['winner']}, expected {want}")

    rules = ['one game, bots seated in the order named' if one else
             'one game per bot, seats rotate so each bot moves first once',
             'turns in seat order, stones_each turns each, last seat moves last',
             'every stone on the board at whole-number coordinates',
             f'every stone at least {min_dist} from every earlier stone',
             'weights are whole numbers >= 1 and within each total',
             'every skipped turn really was an illegal stone',
             'a forfeited bot plays no more that game',
             'clocks stay between 0 and the budget',
             'territory matches a point-by-point recount' + (' (not run)' if skip_recount else ''),
             'totals are summed over the games and the winner has the most']
    print()
    for i, text in enumerate(rules, 1):
        bad = r.failures.get(i)
        print(f"  {'FAIL' if bad else 'PASS'}  {i:2}. {text}")
        for d in (bad or [])[:5]:
            print(f'           {d}')
    skips = sum(1 for g in games for m in g['moves'] if m.get('refused'))
    forfeits = sum(1 for g in games for e in (g.get('errors') or []) if e)
    print(f'\n  {skips} skipped turn(s), {forfeits} forfeit(s) in this match.')
    return not r.failures


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('results', nargs='+', help="a match's results.json (or several)")
    ap.add_argument('--no-recount', action='store_true', help='skip the slow territory recount')
    args = ap.parse_args()
    ok = True
    for path in args.results:
        ok = verify(path, args.no_recount) and ok
        print()
    print('ALL RULES PASS' if ok else 'SOME RULES FAILED')
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
