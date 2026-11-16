"""The rules of the Gravitational Voronoi game: the board, placing stones, and the score.

The board. A GRID x GRID square of integer points, x from 0 (left) to GRID-1 (right) and
y from 0 (top) to GRID-1 (bottom). Stones go on these points and the score counts them.

A game. The players take turns placing one stone each, in seat order (player 1 first),
until every player has had STONES turns, so the last stone is placed by the last player.
A stone is legal when

  * x and y are whole numbers on the board,
  * its Euclidean distance to every stone already placed, of any colour, is at least MIN_DIST,
  * its weight is a whole number, at least 1 and no more than the player's weight left.

An illegal stone is not placed. The player's turn is skipped (that stone is lost) and the
reason is written in the referee's notes.

The score. Every point of the board belongs to the colour with the greatest pull on it:

    pull(c, p) = sum over the stones s of colour c of  weight(s) / distance(s, p)^2

A point holding a stone belongs to that stone's colour (its pull there is infinite). Where
two or more colours tie for the greatest pull, the point belongs to nobody. Pulls are
floating point, so "tie" means equal to within TIE_TOLERANCE, relative. A player's score is
the number of points it owns.
"""
import json
from array import array

GRID = 1000                 # the board is GRID x GRID integer points
MIN_DIST = 66               # minimum Euclidean distance between any two stones
TIE_TOLERANCE = 1e-9        # pulls this close (relative) are a tie: the point belongs to nobody
MAX_PLAYERS = 10            # the most players in one game (a melee)
COLORS = ('red', 'blue', 'green', 'orange', 'purple', 'teal', 'yellow', 'pink', 'brown', 'gray')
FIT_WARNING = 250           # about how many stones fit on the board at MIN_DIST spacing


class Config:
    """The parameters the architect supplies: stones per player, total weight per player."""

    warning = None          # set when the parameters are legal but unusual

    def __init__(self, stones, weight):
        self.stones, self.weight = stones, weight

    def as_dict(self):
        return {'stones': self.stones, 'weight': self.weight}

    def __str__(self):
        return f'stones={self.stones} weight={self.weight}'


def make_config(stones, weight):
    """Check the parameters and return a Config."""
    s, w = whole(stones), whole(weight)
    if s is None or w is None:
        raise ValueError('The number of stones and the weight must be whole numbers.')
    if s < 1:
        raise ValueError('Each player needs at least 1 stone.')
    if w < s:
        raise ValueError(f'The weight per player ({w}) must be at least the number of stones '
                         f'({s}), since every stone weighs at least 1.')
    cfg = Config(s, w)
    if 2 * s > FIT_WARNING:
        cfg.warning = (f'{s} stones each is a lot: only about {FIT_WARNING} stones fit on the '
                       f'board {MIN_DIST} apart, so late stones may have nowhere legal to go.')
    return cfg


def load_config(path):
    """Read a parameter file: two whole numbers (stones per player, weight per player).
    Anything after a # is a comment."""
    nums = []
    with open(path, encoding='utf-8') as f:
        text = f.read()
    for line in text.splitlines():
        line = line.split('#', 1)[0].strip()
        if line:
            nums += line.split()
    if len(nums) != 2:
        raise ValueError(f'{path}: expected two numbers (stones per player, weight per player), '
                         f'found {len(nums)}.')
    return make_config(*nums)


def whole(value):
    """A whole number written any of the ways a JSON encoder might write one, or None.

    A bot whose arithmetic produced a float may send 34.0, and one that builds its reply by
    hand may send "34". Both are read as 34. A real fraction such as 34.5 is not a whole
    number, and true/false are not numbers at all.
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value) if value.is_integer() else None
    if isinstance(value, str):
        try:
            return whole(json.loads(value.strip()))
        except (ValueError, AttributeError):
            return None
    return None


class Game:
    """One game between `players` players, seated 0 .. players-1 in turn order."""

    def __init__(self, config, players=2, grid=GRID, min_dist=MIN_DIST):
        if not 2 <= players <= MAX_PLAYERS:
            raise ValueError(f'A game needs between 2 and {MAX_PLAYERS} players.')
        self.cfg = config
        self.k = players
        self.grid = grid
        self.min_dist = min_dist
        self.stones = []                         # every stone placed, in order: dicts
        self.weight_left = [config.weight] * players
        self.placed = [0] * players              # stones each player has on the board
        self.skipped = [0] * players             # turns each player lost to an illegal stone
        self.turn = 0                            # whose turn it is (a seat number)
        self.turns_done = 0
        self.total_turns = players * config.stones
        self.notes = []
        self.colors = list(COLORS[:players])     # colour of each seat; a match sets it per bot
        g = grid
        zero = bytes(8 * g)
        # pull[p][y][x], the best pull of anyone at each point, and its owner (0 = nobody).
        self.pull = [[array('d', zero) for _ in range(g)] for _ in range(players)]
        self.best = [array('d', zero) for _ in range(g)]
        self.owner = [bytearray(g) for _ in range(g)]
        self._scores = [0] * players

    # -- what a bot is told -------------------------------------------------

    def state(self):
        """The whole position, as sent to the player whose turn it is."""
        p = self.turn
        return {'you': p + 1, 'players': self.k, 'color': self.colors[p],
                'grid': self.grid, 'min_dist': self.min_dist,
                'stones_each': self.cfg.stones, 'weight_each': self.cfg.weight,
                'turn': self.placed[p] + self.skipped[p] + 1,
                'stones': [[s['player'] + 1, s['x'], s['y'], s['weight']] for s in self.stones],
                'stones_left': [self.cfg.stones - self.placed[q] - self.skipped[q]
                                for q in range(self.k)],
                'weight_left': list(self.weight_left),
                'scores': self.scores()}

    # -- a move ---------------------------------------------------------------

    def check(self, player, reply):
        """Is this stone legal? Returns ((x, y, weight), None) or (None, reason)."""
        if not isinstance(reply, dict):
            return None, 'the reply is not a JSON object'
        x, y, w = whole(reply.get('x')), whole(reply.get('y')), whole(reply.get('weight'))
        for name, raw, val in (('x', reply.get('x'), x), ('y', reply.get('y'), y),
                               ('weight', reply.get('weight'), w)):
            if val is None:
                if raw is None:
                    return None, f'the reply has no "{name}"'
                return None, f'{name} = {json.dumps(raw)} is not a whole number'
        if not (0 <= x < self.grid and 0 <= y < self.grid):
            return None, f'({x}, {y}) is off the board (0 .. {self.grid - 1})'
        if w < 1:
            return None, f'weight {w} is not allowed: every stone weighs at least 1'
        if w > self.weight_left[player]:
            return None, f'weight {w} is more than the {self.weight_left[player]} left'
        limit = self.min_dist * self.min_dist
        for s in self.stones:
            d2 = (s['x'] - x) ** 2 + (s['y'] - y) ** 2
            if d2 < limit:
                whose = 'its own' if s['player'] == player else f"{self.colors[s['player']]}'s"
                return None, (f'({x}, {y}) is {d2 ** 0.5:.1f} from {whose} stone at '
                              f"({s['x']}, {s['y']}); stones must be at least {self.min_dist} apart")
        return (x, y, w), None

    def place(self, player, reply):
        """Place a stone for `player`. Returns (stone_or_None, reason_if_refused)."""
        move, why = self.check(player, reply)
        if why:
            self.skipped[player] += 1
            return None, why
        x, y, w = move
        stone = {'player': player, 'x': x, 'y': y, 'weight': w}
        self.stones.append(stone)
        self.weight_left[player] -= w
        self.placed[player] += 1
        self._add_pull(player, x, y, w)
        return stone, None

    def skip(self, player):
        """The player loses this turn without a stone (it forfeited or ran out of time)."""
        self.skipped[player] += 1

    def _add_pull(self, p, sx, sy, w):
        """Add one stone's pull to its player and update who owns every point.

        Only player p's pull goes up, so a point changes hands only where p's new pull beats
        (or ties) the best pull anyone had there. That keeps every move O(points) however
        many players there are.
        """
        g = self.grid
        dx2 = [(x - sx) ** 2 for x in range(g)]
        hi = 1.0 + TIE_TOLERANCE
        q = p + 1
        rows = self.pull[p]
        for y in range(g):
            dy2 = (y - sy) ** 2
            old = rows[y]
            if dy2:
                new = array('d', [a + w / (dy2 + d) for a, d in zip(old, dx2)])
            else:                                # the stone's own row: skip the stone's point
                new = array('d', [a + (w / d if d else 0.0) for a, d in zip(old, dx2)])
            rows[y] = new
            best = self.best[y]
            own = self.owner[y]
            self.owner[y] = bytearray([o if o == q else q if a > b * hi else 0 if a * hi >= b else o
                                       for a, b, o in zip(new, best, own)])
            self.best[y] = array('d', [a if a > b else b for a, b in zip(new, best)])
        for s in self.stones:                    # a stone's own point is always its colour's
            self.owner[s['y']][s['x']] = s['player'] + 1
        self._scores = [sum(row.count(c + 1) for row in self.owner) for c in range(self.k)]

    # -- the score -----------------------------------------------------------

    def scores(self):
        return list(self._scores)

    def owner_at(self, x, y):
        """Who owns point (x, y): a seat number, or None for a tie."""
        o = self.owner[y][x]
        return o - 1 if o else None

    def done(self):
        return self.turns_done >= self.total_turns

    def advance(self):
        """Hand the turn to the next player."""
        self.turns_done += 1
        self.turn = (self.turn + 1) % self.k
