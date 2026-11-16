"""Runs Gravitational Voronoi matches between bots. The referee starts each bot itself.

Bots are run the same way as in the Evasion, Kelly and PegBand architectures: every game
starts a fresh process for each bot, from its folder in bots.json. The bot gets a "start"
message and must reply {"ready": true} within STARTUP_LIMIT seconds, which is not on its
clock. The players then move one at a time, and each is timed from the referee's message
to its answer.

Two kinds of failure, treated differently:

  * An illegal stone (off the board, too close to another stone, a bad weight, a missing
    field). The stone is not placed, that turn is SKIPPED, and the game goes on: the bot is
    asked again on its next turn. Every skip is written to the referee's notes, the match's
    log.txt and the console.
  * A forfeit: the bot ran out of time, crashed, or broke the protocol (a reply that is not
    one line of JSON). It is asked nothing more that game, its remaining turns are skipped,
    and it keeps the territory its stones already hold. The others play on.

Messages are one JSON object per line. See the README, "The messages between referee and bot".
"""
import json
import os
import queue
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .engine import COLORS, Game, load_config, make_config

ROOT = Path(__file__).resolve().parent.parent
PROTOCOL_VERSION = 1
STARTUP_LIMIT = 30.0        # seconds a bot may take to start (not on its clock)
OUTPUT_LIMIT = 1 << 20      # bytes a bot may print per reply (stdout or stderr)
NOTE_LIMIT = 400            # referee notes kept per game


class BotError(Exception):
    """A bot broke the protocol: crash, bad JSON, too much output, or timeout."""


def _resolve(arg, cwd):
    """{python} -> this Python; ./prog -> full path to prog in the bot folder (prog.exe on Windows)."""
    if arg == '{python}':
        return sys.executable
    if arg.startswith(('./', '.\\')):
        p = Path(cwd) / arg[2:]
        if os.name == 'nt' and not p.suffix:
            p = p.with_suffix('.exe')
        return str(p)
    return arg


def load_bots(path='bots.json'):
    """Read the roster. Each entry: name, cwd, command, optional build.

    bots.local.json next to it, if present, is read too and its entries added: it is ignored
    by git, so classmates' bots can be listed there without ever being pushed.
    """
    path = (ROOT / path).resolve()
    entries = json.loads(path.read_text(encoding='utf-8'))
    local = path.with_name(path.stem + '.local.json')
    if local.exists():
        extra = json.loads(local.read_text(encoding='utf-8'))
        if not isinstance(extra, list):
            raise ValueError(f'{local.name} must be a list of bots')
        entries = list(entries) + extra
    if not isinstance(entries, list):
        raise ValueError('bots.json must be a list of bots')
    entries = list(entries) + discover_bots(path.parent, entries)
    names = set()
    for bot in entries:
        name = bot.get('name')
        if not isinstance(name, str) or not name.strip() or name in names:
            raise ValueError('Every bot needs a unique, non-empty "name"')
        names.add(name)
        cwd = (path.parent / bot.get('cwd', '.')).resolve()
        if not cwd.is_dir():
            raise ValueError(f'{name}: folder not found: {cwd}')
        for key in ('command', 'build'):
            cmd = bot.get(key)
            if key == 'build' and cmd is None:
                continue
            if not isinstance(cmd, list) or not cmd or not all(isinstance(x, str) for x in cmd):
                raise ValueError(f'{name}: "{key}" must be a non-empty list of strings')
            bot[key] = [_resolve(x, cwd) for x in cmd]
        bot['cwd'] = str(cwd)
    return entries


def discover_bots(root, listed):
    """Bots dropped into bots/ without a roster entry: any folder (other than templates/)
    holding a bot.py, bot.jl or bot.cpp is registered under the folder's name.

    This is how a classmate's bot is added: unzip their folder into bots/ and press Rescan.
    """
    root = Path(root)
    bots_dir = root / 'bots'
    if not bots_dir.is_dir():
        return []
    taken_dirs = {str((root / b.get('cwd', '.')).resolve()) for b in listed if isinstance(b, dict)}
    taken_names = {b.get('name') for b in listed if isinstance(b, dict)}
    found = []
    for folder in sorted(p for p in bots_dir.rglob('*') if p.is_dir()):
        rel_parts = folder.relative_to(bots_dir).parts
        if 'templates' in rel_parts or any(x.startswith(('.', '_')) for x in rel_parts):
            continue
        if str(folder.resolve()) in taken_dirs:
            continue
        rel = folder.relative_to(root).as_posix()
        if (folder / 'bot.py').is_file():
            entry = {'name': folder.name, 'cwd': rel, 'command': ['{python}', 'bot.py']}
        elif (folder / 'bot.jl').is_file():
            entry = {'name': folder.name, 'cwd': rel,
                     'command': ['julia', '-t', 'auto', '--startup-file=no', 'bot.jl']}
        elif (folder / 'bot.cpp').is_file():
            entry = {'name': folder.name, 'cwd': rel, 'command': ['./bot'],
                     'build': ['{python}', 'build.py'] if (folder / 'build.py').is_file()
                     else ['c++', '-std=c++17', '-O2', 'bot.cpp', '-o', 'bot']}
        else:
            continue
        if entry['name'] in taken_names:
            entry['name'] = f"{entry['name']} ({rel})"
        found.append(entry)
        taken_names.add(entry['name'])
        taken_dirs.add(str(folder.resolve()))
    return found


_built = {}          # bot folder -> fingerprint of its files right after the last successful build
_build_errors = {}   # bot folder -> why its last build failed, so starting it says so


def _fingerprint(folder):
    out = []
    for f in sorted(Path(folder).rglob('*')):
        if f.is_file() and '__pycache__' not in f.parts:
            st = f.stat()
            out.append((str(f), st.st_mtime_ns, st.st_size))
    return hash(tuple(out))


def build(bot):
    """Run a bot's build step (e.g. compiling C++) outside any clock.

    A failed build also throws away whatever the last successful build left behind, so a bot
    whose source no longer compiles cannot quietly keep playing with an older binary. The
    error is remembered against the folder, and starting that bot re-raises it, so the
    failure shows up as that bot's forfeit instead of disappearing into the log.
    """
    if not bot.get('build'):
        return
    key = str(Path(bot['cwd']).resolve())
    if _built.get(key) == _fingerprint(key):
        return
    _built.pop(key, None)
    try:
        r = subprocess.run(bot['build'], cwd=bot['cwd'], capture_output=True, text=True, timeout=300)
        failed = r.returncode != 0
        why = f"build failed: {(r.stderr or r.stdout).strip()[-300:]}" if failed else ''
    except (OSError, subprocess.TimeoutExpired) as exc:
        failed, why = True, f'build failed: {exc}'
    if failed:
        _build_errors[key] = why
        for arg in bot['command']:                 # drop the stale product of an earlier build
            target = Path(arg)
            if target.is_file() and target.parent == Path(bot['cwd']).resolve():
                try:
                    target.unlink()
                except OSError:
                    pass
        raise RuntimeError(why)
    _build_errors.pop(key, None)
    _built[key] = _fingerprint(key)


class BotSession:
    """One running bot process for one game."""

    def __init__(self, bot):
        self.bot = bot
        self.proc = None
        self.lines = queue.Queue()
        self.stderr_tail = []
        self.stderr_bytes = 0
        self.stderr_dropped = 0      # stderr already accounted for by an earlier reply
        self.readers = []

    def start(self):
        failed = _build_errors.get(str(Path(self.bot['cwd']).resolve()))
        if failed:
            raise BotError(failed)
        try:
            self.proc = subprocess.Popen(self.bot['command'], cwd=self.bot['cwd'],
                                         stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                         stderr=subprocess.PIPE, text=True, bufsize=1,
                                         encoding='utf-8', errors='replace')
        except OSError as exc:
            raise BotError(f'could not start: {exc}')
        self.readers = [threading.Thread(target=self._read_out, daemon=True),
                        threading.Thread(target=self._read_err, daemon=True)]
        for r in self.readers:
            r.start()

    def _read_out(self):
        for line in self.proc.stdout:
            self.lines.put(line)
        self.lines.put(None)

    def _read_err(self):
        """Drain the bot's standard error, counting it but keeping only the last few lines.

        Read in fixed-size chunks, not by line, because a bot that prints megabytes without a
        newline would otherwise sit in the reader's buffer uncounted. The tail is what an
        error message quotes.
        """
        while True:
            chunk = self.proc.stderr.read(8192)
            if not chunk:
                break
            self.stderr_bytes += len(chunk)
            if self.stderr_bytes - self.stderr_dropped <= OUTPUT_LIMIT:
                self.stderr_tail = [(''.join(self.stderr_tail) + chunk)[-800:]]

    def ask(self, msg, timeout):
        """Send one message, wait for one JSON reply. Returns (reply, seconds)."""
        start = time.monotonic()
        before_err = self.stderr_bytes
        try:
            self.proc.stdin.write(json.dumps(msg, separators=(',', ':')) + '\n')
            self.proc.stdin.flush()
        except (OSError, ValueError):
            raise BotError('crashed' + self._err_note())
        try:
            line = self.lines.get(timeout=max(0.001, timeout))
        except queue.Empty:
            raise BotError('time')
        elapsed = time.monotonic() - start
        if line is None:
            raise BotError('crashed' + self._err_note())
        too_much = len(line) > OUTPUT_LIMIT or self.stderr_bytes - before_err > OUTPUT_LIMIT
        self.stderr_dropped = self.stderr_bytes
        if too_much:
            raise BotError('too much output')
        try:
            reply = json.loads(line)
        except ValueError:
            raise BotError(f'invalid JSON: {line.strip()[:60]!r}')
        if not isinstance(reply, dict):
            raise BotError('reply must be a JSON object')
        return reply, elapsed

    def _err_note(self):
        tail = ''.join(self.stderr_tail).strip()
        return f' ({tail[-160:]})' if tail else ''

    def close(self):
        if not self.proc:
            return
        try:
            self.proc.stdin.close()
        except OSError:
            pass
        try:
            self.proc.wait(timeout=1)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            self.proc.wait()
        for r in self.readers:
            r.join(timeout=1)
        for stream in (self.proc.stdout, self.proc.stderr):
            try:
                stream.close()
            except OSError:
                pass


def play_game(bots, config, clock=120.0, game=1, on_move=None, delay=0, stop=None, live=None,
              log=None, game_factory=Game, colors=None):
    """Play one game. bots[i] sits in seat i; seat 0 moves first, the last seat moves last.
    colors[i] is the colour of seat i (by default red, blue, ... in seat order).

    Returns a dict with the scores, every turn (for the replay), the referee's notes, the
    errors and the clocks. live (optional dict) is kept up to date for the dashboard, and
    log (optional function) is called with every note as it happens.
    """
    k = len(bots)
    g = game_factory(config, players=k)
    if colors:
        g.colors = list(colors)
    clocks = [float(clock)] * k
    if live is not None:
        live['clocks'], live['since'] = clocks, [None] * k
    errors = [None] * k              # once set, that bot is no longer asked anything
    moves = []                       # every turn, in order, for the replay
    notes = []
    sessions = [BotSession(b) for b in bots]
    pool = ThreadPoolExecutor(max_workers=k)
    who = [f"{g.colors[p]} ({bots[p]['name']})" for p in range(k)]

    def note(text):
        if len(notes) < NOTE_LIMIT:
            notes.append(text)
        g.notes.append(text)
        if log:
            log(f'game {game}: {text}')

    try:
        def start(player):           # untimed: launch + start message + ready reply
            s = sessions[player]
            s.start()
            reply, _ = s.ask({'type': 'start', 'protocol_version': PROTOCOL_VERSION,
                              'you': player + 1, 'players': k, 'color': g.colors[player],
                              'grid': g.grid, 'min_dist': g.min_dist,
                              'stones': config.stones, 'weight': config.weight,
                              'clock': clock}, STARTUP_LIMIT)
            if reply.get('ready') is not True:
                raise BotError('did not reply {"ready": true} to the start message')

        for player, fut in enumerate([pool.submit(start, p) for p in range(k)]):
            try:
                fut.result()
            except BotError as exc:
                errors[player] = f'startup: {exc}'
                note(f'FORFEIT: {who[player]} failed to start: {exc}')

        while not g.done():
            if stop is not None and stop.is_set():
                break
            player = g.turn
            turn_no = g.placed[player] + g.skipped[player] + 1
            entry = {'turn': g.turns_done + 1, 'player': player, 'round': turn_no}

            if not errors[player] and clocks[player] <= 0:
                errors[player] = 'time'
                note(f'FORFEIT: {who[player]} ran out of time')
            if errors[player]:                       # forfeited: lose the turn, ask nothing
                g.skip(player)
                entry.update(stone=None, forfeit=errors[player])
            else:
                msg = {'type': 'move', 'clock': round(clocks[player], 3)}
                msg.update(g.state())
                if live is not None:
                    live['since'][player] = time.monotonic()
                try:
                    reply, elapsed = sessions[player].ask(msg, clocks[player])
                    clocks[player] = max(0.0, clocks[player] - elapsed)
                    if clocks[player] == 0.0:
                        raise BotError('time')
                except BotError as exc:
                    if str(exc) == 'time':
                        clocks[player] = 0.0
                    errors[player] = str(exc)
                    note(f'FORFEIT: {who[player]} on turn {turn_no}: '
                         f'{"ran out of time" if str(exc) == "time" else exc}. '
                         f'Its remaining turns are skipped.')
                    reply = None
                finally:
                    if live is not None:
                        live['since'][player] = None
                if reply is None:
                    g.skip(player)
                    entry.update(stone=None, forfeit=errors[player])
                else:
                    stone, why = g.place(player, reply)
                    entry['stone'] = stone
                    if why:
                        entry['refused'] = why
                        entry['sent'] = {k2: reply.get(k2) for k2 in ('x', 'y', 'weight')}
                        note(f'SKIPPED: {who[player]} turn {turn_no}: {why}. '
                             f'No stone placed; the turn is skipped.')
            entry['clocks'] = [round(c, 2) for c in clocks]
            entry['scores'] = g.scores()
            entry['weight_left'] = list(g.weight_left)
            moves.append(entry)
            g.advance()
            if on_move:
                on_move(g, entry, clocks)
            d = delay() if callable(delay) else delay
            if d and not g.done():
                if stop is not None:
                    stop.wait(d)
                else:
                    time.sleep(d)

        return {'scores': g.scores(), 'moves': moves, 'notes': notes, 'errors': errors,
                'clocks': clocks, 'skipped': list(g.skipped), 'placed': list(g.placed),
                'stones': [[s['player'], s['x'], s['y'], s['weight']] for s in g.stones],
                'grid': g.grid, 'min_dist': g.min_dist,
                'ended': 'finished' if g.done() else 'stopped'}
    finally:
        for s in sessions:
            s.close()
        pool.shutdown(wait=False)


def resolve_config(config):
    """A Config from a parameter file path, a dict typed into the page, or a Config."""
    if isinstance(config, dict):
        cfg = make_config(config.get('stones'), config.get('weight'))
        return cfg, str(cfg)
    if isinstance(config, (str, Path)):
        return load_config(ROOT / config), str(config)
    return config, str(config)


def play_match(bots, config, clock=120.0, out_dir=None, on_progress=None, delay=0, stop=None,
               live=None, echo=False, game_factory=Game, one_game=False):
    """One match between two or more bots: as many games as there are bots.

    The seating rotates so that every bot sits in every seat exactly once: in game i, bot i
    moves first. Each bot keeps one colour for the whole match (the first bot named is red,
    the second blue, ...), whichever seat it is in. With two bots that is the spec's "your score as first player plus your score
    as second player"; with more (a melee) it is the same idea. A bot's match score is its
    territory summed over all the games, and the highest total wins.

    one_game=True plays just the first game instead: the bots sit in the order given, so the
    first bot moves first, and the match score is that one game's territory.

    Writes results.json, replay.html and log.txt to out_dir (if given) and returns the state.
    echo=True also prints every referee note (skips, forfeits) to the console.
    """
    cfg, label = resolve_config(config)
    k = len(bots)
    if k < 2:
        raise ValueError('A match needs at least two bots.')
    names = [b['name'] for b in bots]
    if len(set(names)) != k:
        raise ValueError('A bot may appear only once in a match.')
    n_games = 1 if one_game else k
    state = {'status': 'Starting', 'names': names, 'config': label, 'params': cfg.as_dict(),
             'time_budget': clock, 'players': k, 'colors': list(COLORS[:k]), 'games': [],
             'one_game': bool(one_game)}
    if cfg.warning:
        state['warning'] = cfg.warning
    totals = [0] * k
    log_lines = []

    def log(text):
        line = f'{time.strftime("%H:%M:%S")}  {text}'
        log_lines.append(line)
        if echo:
            print(f'    {text}', flush=True)

    log(f"match: {' vs '.join(names)}  ({label}, {clock:g}s per bot per game"
        f"{', one game only' if one_game else ''})")
    for game in range(1, n_games + 1):
        order = [(game - 1 + s) % k for s in range(k)]      # order[seat] = index into bots
        seated = [bots[i] for i in order]
        colors = [COLORS[i] for i in order]                 # each bot keeps its own colour
        gstate = {'game': game, 'seats': [b['name'] for b in seated], 'order': order,
                  'colors': colors,
                  'status': 'starting the bots', 'scores': [0] * k}
        state['games'].append(gstate)
        state['status'] = f'Game {game} of {n_games} · starting the bots'
        log(f"game {game}: seats " + ', '.join(f'{colors[s]}={n}' for s, n in enumerate(gstate['seats'])))
        if on_progress:
            on_progress(state)
        if live is not None:
            live.clear()
            live['game'] = game

        def on_move(g, move, clocks, gstate=gstate, game=game):
            gstate['status'] = 'playing'
            gstate['scores'] = g.scores()
            gstate['turn'] = g.turns_done
            gstate['total_turns'] = g.total_turns
            state['status'] = f'Game {game} of {n_games} · turn {g.turns_done} of {g.total_turns}'
            if on_progress:
                on_progress(state, g)

        result = play_game(seated, cfg, clock, game, on_move=on_move, delay=delay, stop=stop,
                           live=live, log=log, game_factory=game_factory, colors=colors)
        gstate.update(result, status='finished' if result['ended'] == 'finished' else 'stopped')
        state['grid'], state['min_dist'] = result['grid'], result['min_dist']
        for seat, i in enumerate(order):
            totals[i] += result['scores'][seat]
        log(f"game {game}: final " + ', '.join(f'{n} {s}' for n, s in zip(gstate['seats'], result['scores'])))
        state['status'] = f'Game {game} finished'
        if on_progress:
            on_progress(state)
        if stop is not None and stop.is_set():
            break

    complete = len(state['games']) == n_games and all(gs['status'] == 'finished' for gs in state['games'])
    best = max(totals)
    winner = None
    if complete and totals.count(best) == 1:
        winner = names[totals.index(best)]
    state['status'] = 'Finished' if complete else 'Stopped'
    state['summary'] = {
        'type': 'match_over', 'totals': dict(zip(names, totals)), 'winner': winner,
        'games': [{'game': gs['game'], 'seats': gs['seats'], 'colors': gs.get('colors'),
                   'scores': gs.get('scores'),
                   'skipped': gs.get('skipped'),
                   'errors': gs.get('errors') if any(gs.get('errors') or []) else None}
                  for gs in state['games']]}
    log('totals: ' + ', '.join(f'{n} {t}' for n, t in zip(names, totals)) +
        f"  ->  winner: {winner or 'tie' if complete else 'none (stopped)'}")
    if out_dir:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(state, separators=(',', ':'))
        (out / 'results.json').write_text(payload, encoding='utf-8')
        (out / 'log.txt').write_text('\n'.join(log_lines) + '\n', encoding='utf-8')
        html = (ROOT / 'dashboard' / 'replay.html').read_text(encoding='utf-8')
        (out / 'replay.html').write_text(
            html.replace('/* REPLAY_DATA */ null', payload.replace('<', '\\u003c')), encoding='utf-8')
    return state
