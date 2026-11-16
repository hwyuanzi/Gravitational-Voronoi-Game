"""Checks that the referee handles bots that misbehave: illegal stones, crashes, timeouts,
bad JSON. Each test writes a small bot into a temporary folder and plays a real game against
it, on a small board so the tests run quickly.
"""
import json
import sys
import tempfile
import unittest
from pathlib import Path

from referee.engine import Game, make_config
from referee.runner import discover_bots, play_game, play_match

# Puts stones on a fixed lattice, in order, skipping points that are taken or too close.
GOOD = """
import json, sys
for line in sys.stdin:
    m = json.loads(line)
    if m['type'] == 'start':
        print(json.dumps({'ready': True}), flush=True); continue
    me = m['you'] - 1
    w = m['weight_left'][me] // m['stones_left'][me]
    g, d = m['grid'], m['min_dist']
    for x in range(0, g, d):
        for y in range(0, g, d):
            if all((x - a) ** 2 + (y - b) ** 2 >= d * d for _, a, b, _ in m['stones']):
                break
        else:
            continue
        break
    print(json.dumps({'x': x, 'y': y, 'weight': w}), flush=True)
"""


def small(cfg, players):
    return Game(cfg, players=players, grid=60, min_dist=8)


def scripted(replies):
    """A bot that answers its moves with these replies, in order (the last one repeats)."""
    return f"""
import json, sys
replies = {json.dumps(replies)}
n = 0
for line in sys.stdin:
    m = json.loads(line)
    if m['type'] == 'start':
        print(json.dumps({{'ready': True}}), flush=True); continue
    r = replies[min(n, len(replies) - 1)]; n += 1
    print(r if isinstance(r, str) else json.dumps(r), flush=True)
"""


class Misbehaving(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.cfg = make_config(3, 9)

    def tearDown(self):
        self.dir.cleanup()

    def bot(self, name, source):
        folder = Path(self.dir.name) / f'{name}{len(list(Path(self.dir.name).iterdir()))}'
        folder.mkdir()
        (folder / 'bot.py').write_text(source, encoding='utf-8')
        return {'name': name, 'cwd': str(folder), 'command': [sys.executable, 'bot.py']}

    def play(self, bots, clock=20.0):
        logged = []
        r = play_game(bots, self.cfg, clock, game_factory=small, log=logged.append)
        r['logged'] = logged
        return r

    def test_two_good_bots_place_every_stone(self):
        r = self.play([self.bot('a', GOOD), self.bot('b', GOOD)])
        self.assertEqual(r['placed'], [3, 3])
        self.assertEqual(r['skipped'], [0, 0])
        self.assertEqual(len(r['moves']), 6)
        self.assertEqual([m['player'] for m in r['moves']], [0, 1, 0, 1, 0, 1])
        self.assertEqual(sum(r['scores']) <= 60 * 60, True)

    def test_an_illegal_stone_skips_that_turn_only_and_is_logged(self):
        bad = scripted([{'x': 1, 'y': 1, 'weight': -5},          # negative weight
                        {'x': 999, 'y': 1, 'weight': 1},         # off the board
                        {'x': 30, 'y': 30, 'weight': 1}])        # fine
        r = self.play([self.bot('good', GOOD), self.bot('bad', bad)])
        self.assertEqual(r['placed'], [3, 1])
        self.assertEqual(r['skipped'], [0, 2])
        self.assertEqual(r['errors'], [None, None])              # skipped, not forfeited
        refused = [m for m in r['moves'] if m.get('refused')]
        self.assertEqual(len(refused), 2)
        self.assertIn('at least 1', refused[0]['refused'])
        self.assertIn('off the board', refused[1]['refused'])
        self.assertEqual(refused[0]['sent'], {'x': 1, 'y': 1, 'weight': -5})
        skips = [n for n in r['logged'] if 'SKIPPED' in n]
        self.assertEqual(len(skips), 2)
        self.assertIn('blue (bad)', skips[0])

    def test_a_bot_that_is_always_illegal_cannot_stall_the_game(self):
        r = self.play([self.bot('good', GOOD), self.bot('bad', scripted([{'x': 'left'}]))])
        self.assertEqual(r['ended'], 'finished')
        self.assertEqual(r['placed'], [3, 0])
        self.assertEqual(r['skipped'], [0, 3])

    def test_too_close_and_fractional_weights_are_skipped(self):
        r = self.play([self.bot('good', GOOD),
                       self.bot('bad', scripted([{'x': 2, 'y': 2, 'weight': 1},
                                                 {'x': 40, 'y': 40, 'weight': 1.5}]))])
        reasons = [m['refused'] for m in r['moves'] if m.get('refused')]
        self.assertIn('apart', reasons[0])
        self.assertIn('not a whole number', reasons[1])

    def test_a_crash_forfeits_and_the_other_player_plays_on(self):
        crash = """
import json, sys
for line in sys.stdin:
    m = json.loads(line)
    if m['type'] == 'start':
        print(json.dumps({'ready': True}), flush=True); continue
    raise SystemExit(1)
"""
        r = self.play([self.bot('crash', crash), self.bot('good', GOOD)])
        self.assertTrue(r['errors'][0].startswith('crashed'))
        self.assertEqual(r['placed'], [0, 3])
        self.assertTrue(any('FORFEIT' in n for n in r['logged']))

    def test_running_out_of_time_forfeits_and_a_late_stone_is_not_placed(self):
        slow = """
import json, sys, time
for line in sys.stdin:
    m = json.loads(line)
    if m['type'] == 'start':
        print(json.dumps({'ready': True}), flush=True); continue
    time.sleep(3)
    print(json.dumps({'x': 30, 'y': 30, 'weight': 1}), flush=True)
"""
        r = self.play([self.bot('good', GOOD), self.bot('slow', slow)], clock=1.0)
        self.assertEqual(r['errors'][1], 'time')
        self.assertEqual(r['placed'][1], 0)
        self.assertEqual(r['clocks'][1], 0.0)
        self.assertEqual(r['placed'][0], 3)

    def test_bad_json_forfeits(self):
        r = self.play([self.bot('good', GOOD), self.bot('chatty', scripted(['hello there']))])
        self.assertTrue(r['errors'][1].startswith('invalid JSON'))

    def test_a_bot_that_never_says_ready_forfeits_at_startup(self):
        mute = "import sys\nfor line in sys.stdin:\n    print('{}', flush=True)\n"
        r = self.play([self.bot('mute', mute), self.bot('good', GOOD)])
        self.assertTrue(r['errors'][0].startswith('startup'))
        self.assertEqual(r['placed'], [0, 3])


class Matches(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.dir.cleanup()

    def bot(self, name, source=GOOD):
        folder = Path(self.dir.name) / name
        folder.mkdir()
        (folder / 'bot.py').write_text(source, encoding='utf-8')
        return {'name': name, 'cwd': str(folder), 'command': [sys.executable, 'bot.py']}

    def test_seats_rotate_so_everyone_moves_first_once_and_totals_add_up(self):
        bots = [self.bot(n) for n in ('a', 'b', 'c')]
        out = Path(self.dir.name) / 'out'
        st = play_match(bots, make_config(2, 4), clock=20, out_dir=out, game_factory=small)
        seats = [g['seats'] for g in st['games']]
        self.assertEqual(seats, [['a', 'b', 'c'], ['b', 'c', 'a'], ['c', 'a', 'b']])
        # each bot keeps its colour whichever seat it is in: a red, b blue, c green
        for g in st['games']:
            self.assertEqual(dict(zip(g['seats'], g['colors'])),
                             {'a': 'red', 'b': 'blue', 'c': 'green'})
        totals = {n: 0 for n in 'abc'}
        for g in st['games']:
            for n, s in zip(g['seats'], g['scores']):
                totals[n] += s
        self.assertEqual(st['summary']['totals'], totals)
        for f in ('results.json', 'replay.html', 'log.txt'):
            self.assertTrue((out / f).is_file(), f)
        self.assertNotIn('/* REPLAY_DATA */ null', (out / 'replay.html').read_text())

    def test_one_game_plays_once_in_the_order_given(self):
        bots = [self.bot(n) for n in ('a', 'b', 'c')]
        st = play_match(bots, make_config(2, 4), clock=20, game_factory=small, one_game=True)
        self.assertEqual(len(st['games']), 1)
        self.assertEqual(st['games'][0]['seats'], ['a', 'b', 'c'])
        self.assertEqual(st['status'], 'Finished')
        self.assertEqual(st['summary']['totals'],
                         dict(zip(['a', 'b', 'c'], st['games'][0]['scores'])))
        self.assertTrue(st['one_game'])

    def test_discovery_finds_a_dropped_in_folder(self):
        root = Path(self.dir.name)
        (root / 'bots' / 'alice').mkdir(parents=True)
        (root / 'bots' / 'alice' / 'bot.py').write_text('')
        (root / 'bots' / 'templates' / 'python').mkdir(parents=True)
        (root / 'bots' / 'templates' / 'python' / 'bot.py').write_text('')
        found = discover_bots(root, [])
        self.assertEqual([b['name'] for b in found], ['alice'])


if __name__ == '__main__':
    unittest.main()
