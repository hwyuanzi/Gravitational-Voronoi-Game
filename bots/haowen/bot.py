"""Gravitational Voronoi protocol v1, Python 3.8+ with no dependencies."""

import json
import math
import sys
from collections import namedtuple
from time import monotonic


Choice = namedtuple('Choice', 'point weight value priority')
State = namedtuple('State', 'margin allowed turns reserve actor')


def distance_sq(a, b):
    dx, dy = a[0] - b[0], a[1] - b[1]
    return dx * dx + dy * dy


def mix(value):
    value = ((value ^ (value >> 16)) * 0x7feb352d) & 0xffffffff
    value = ((value ^ (value >> 15)) * 0x846ca68b) & 0xffffffff
    return value ^ (value >> 16)


def samples(size, side):
    side = min(side, size)
    points = []
    for row in range(side):
        a = min(row, side - 1 - row)
        for col in range(side):
            b = min(col, side - 1 - col)
            jx = 0.13 + 0.74 * (mix(1 + b + 113 * a + 1093 * side) % 1024) / 1024
            jy = 0.13 + 0.74 * (mix(1 + a + 113 * b + 1093 * side) % 1024) / 1024
            x, y = int((b + jx) * size / side), int((a + jy) * size / side)
            points.append((x if col == b else size - 1 - x,
                           y if row == a else size - 1 - y))
    return points


def lattice(size, divisions):
    axis = sorted({(i * (size - 1) + divisions // 2) // divisions
                   for i in range(divisions + 1)})
    return [(x, y) for y in axis for x in axis]


def ring(point, radius, count):
    x, y = point
    for i in range(count):
        angle = math.tau * i / count
        dx, dy = round(radius * math.cos(angle)), round(radius * math.sin(angle))
        if dx * dx + dy * dy < radius * radius:
            if abs(dx) >= abs(dy):
                dx += 1 if dx >= 0 else -1
            else:
                dy += 1 if dy >= 0 else -1
        yield x + dx, y + dy


def weights(left, turns, detail, style=0):
    limit = left - turns + 1
    if turns <= 1 or limit <= 1:
        return (max(1, limit),)
    average = left / turns
    if turns == 2:
        fractions = ((0.0, 0.15, 0.25, 0.38, 0.50, 0.62, 0.75, 0.85, 1.0),
                     (0.12, 0.33, 0.50, 0.67, 0.88), (0.28, 0.52, 0.77))[detail]
        proposed = [left * f for f in fractions]
    elif detail == 0:
        proposed = [average * f for f in (0.42, 0.62, 0.82, 1, 1.22, 1.48, 1.8, 2.15)]
        if turns <= 4:
            proposed.extend((1, 0.68 * left))
    elif detail == 1:
        proposed = [average * f for f in (0.55, 0.85, 1.15, 1.55)]
        if turns == 3:
            proposed.append(0.68 * left)
    else:
        scales = ((0.72, 1.13), (0.63, 0.93), (1.03, 1.55))[style]
        proposed = [average * f for f in scales]
    return sorted({max(1, min(limit, int(w + 0.5))) for w in proposed})


class Deadline:
    def __init__(self, clock, turns):
        slack = max(0.11, min(0.75, 0.065 * clock))
        share = clock / turns
        budget = min(clock - slack, share * (1.4 if turns <= 2 else 0.92), 11.0)
        self.end = monotonic() + max(0.0, budget)

    def left(self):
        return self.end - monotonic()

    def check(self, guard=0.012):
        if self.left() <= guard:
            raise SearchExpired


class SearchExpired(Exception):
    pass


class Board:
    def __init__(self, msg):
        self.size = msg['grid']
        self.spacing = msg['min_dist']
        self.radius_sq = self.spacing * self.spacing
        self.me = msg['you'] - 1
        self.players = msg['players']
        self.total = msg['weight_each']
        self.stones_each = msg['stones_each']
        self.turns = tuple(msg['stones_left'])
        self.reserve = tuple(msg['weight_left'])
        self.stones = tuple((player - 1, (x, y), weight) for player, x, y, weight in msg['stones'])
        self.occupied = tuple(point for _, point, _ in self.stones)

    def legal(self, point):
        x, y = point
        return (0 <= x < self.size and 0 <= y < self.size and
                all(distance_sq(point, other) >= self.radius_sq for other in self.occupied))

    def fallback(self):
        if not self.occupied:
            return self.size // 2, self.size // 2
        point = max(lattice(self.size, 15),
                    key=lambda p: min(distance_sq(p, other) for other in self.occupied))
        if self.legal(point):
            return point
        # A lattice can miss a narrow gap. Skip the forbidden intervals on each scanline.
        for y in range(self.size):
            x = 0
            while x < self.size:
                next_x = x
                for sx, sy in self.occupied:
                    dy = y - sy
                    if (x - sx) ** 2 + dy * dy < self.radius_sq:
                        next_x = max(next_x, sx + math.isqrt(self.radius_sq - 1 - dy * dy) + 1)
                if next_x == x:
                    return x, y
                x = next_x
        return None


class Field:
    """Fixed-point pull at stratified samples, stored in lanes of a Python integer.

    Each lane has a guard bit. Addition and multiplication by a stone's weight
    update all samples together. Guard bits count wins or clamp the margins in
    parallel. Taking the remainder modulo 2**bits - 1 adds the clipped lanes.
    """

    def __init__(self, size, side, total, forecast=True):
        self.points = samples(size, side)
        self.count = len(self.points)
        self.area = size * size
        self.scale = 256
        self.cap = 1024 * self.scale
        needed = max(8 * total * self.cap, 8 * self.count * total * self.scale)
        self.bits = max(24, 8 * ((needed.bit_length() + 8) // 8))
        self.modulus = (1 << self.bits) - 1
        self.ones = ((1 << (self.count * self.bits)) - 1) // self.modulus
        self.shift = self.bits - 1
        self.guard = self.ones << self.shift
        self.raw = {}
        self.kernels = {}
        self.future = 0
        if forecast:
            def background(x, y):
                px, py = (x + 0.5) / size, (y + 0.5) / size
                return sum(1.0 / ((px - (col + 0.5) / 10) ** 2 +
                                  (py - (row + 0.5) / 10) ** 2 + 0.016)
                           for row in range(10) for col in range(10)) / 100

            self.future = self.pack(round(self.scale * background(x, y)) for x, y in self.points)

    def pack(self, values):
        # Byte-aligned lanes make packing linear in the number of samples.
        lane = self.bits // 8
        return int.from_bytes(b''.join(value.to_bytes(lane, 'little') for value in values), 'little')

    def pull(self, point):
        cached = self.raw.get(point)
        if cached is not None:
            return cached
        x, y = point
        unit = self.area * self.scale
        cap = self.cap
        limit = unit // cap
        result = self.pack(cap if d2 <= limit else (unit + d2 // 2) // d2
                           for d2 in ((x - sx) * (x - sx) + (y - sy) * (y - sy)
                                      for sx, sy in self.points))
        self.raw[point] = result
        return result

    def kernel(self, point):
        result = self.kernels.get(point)
        if result is None:
            result = self.pull(point) - self.future
            self.kernels[point] = result
        return result

    def control(self, margin, span):
        # Bias signed margins into the guard-bit range. Keep the interior lanes,
        # replace those above the upper bound, and zero those below the lower bound.
        biased = self.guard + span + margin
        high = (biased - 2 * span) & self.guard
        middle = ((biased & self.guard) ^ high) >> self.shift
        clipped = ((biased ^ self.guard) & (middle * self.modulus))
        clipped |= (high >> self.shift) * (2 * (span & self.modulus))
        return clipped % self.modulus

    def width(self, weight, total):
        return max(1, round(self.scale * (0.11 * total + 1.75 * weight)))

    def wins(self, margin):
        flags = (self.guard + margin - self.ones) & self.guard
        try:
            return flags.bit_count()
        except AttributeError:  # Python 3.8/3.9
            return bin(flags).count('1')


class PairSearch:
    def __init__(self, board, field, fast_field, points, deadline):
        self.board = board
        self.field = field
        self.fast_field = fast_field
        self.deadline = deadline
        self.points = tuple(points)
        self.masks = {}
        self.root_ids = tuple(range(len(points)))
        size = board.size
        self.reply_ids = tuple(i for i in range(len(points)) if i % 3 != 1)
        self.rollout_ids = tuple(tuple(i for i, (x, y) in enumerate(points)
                                       if (x * 19 + y * 11 + i * (2 + style)) % 17 < 3)
                                 for style in range(2))
        self.diversity = max(1, size // 20) ** 2
        self.root = self.initial(field)
        self.fast_root = self.initial(fast_field)

    def initial(self, field):
        me = self.board.me
        reserve = tuple(w if t > 0 else 0 for t, w in zip(self.board.turns, self.board.reserve))
        turns = tuple(min(t, w) for t, w in zip(self.board.turns, reserve))
        margin = (reserve[me] - reserve[1 - me]) * field.future
        for player, point, weight in self.board.stones:
            margin += (weight if player == me else -weight) * field.pull(point)
        return State(margin, (1 << len(self.points)) - 1, turns, reserve, me)

    def legal_mask(self, index):
        mask = self.masks.get(index)
        if mask is None:
            point = self.points[index]
            blocked = sum(1 << i for i, other in enumerate(self.points)
                          if i == index or distance_sq(point, other) < self.board.radius_sq)
            mask = ~blocked
            self.masks[index] = mask
        return mask

    def rank(self, state, detail, count, field=None, style=0, layout=0, immediate=False):
        self.deadline.check()
        field = field or self.field
        actor = state.actor
        left, turns = state.reserve[actor], state.turns[actor]
        if not turns:
            return []
        if detail == 0:
            indices = self.root_ids
        elif detail == 1:
            indices = self.reply_ids
        else:
            indices = self.rollout_ids[layout]
        if immediate:
            allocations = (max(1, left // turns),)
        else:
            allocations = weights(left, turns, detail, style)
        targets = (0.92, 0.72, 1.18)
        tests = []
        for weight in allocations:
            unspent = left - weight + state.reserve[1 - actor]
            width = max(1, round(0.12 * field.scale * self.board.total)) if immediate else field.width(
                unspent, self.board.total)
            deviation = min(2.0, abs(weight * turns / left - targets[style]))
            prior = 0.016 * deviation * deviation if turns > 2 and not immediate else 0.0
            margin = state.margin
            if immediate:
                margin -= (left - weight - state.reserve[1 - actor]) * field.future
            tests.append((weight, width * field.ones, 1 / (width * field.count), prior, margin))
        terminal = sum(state.turns) == 1 and not immediate
        ranking = []
        best = None
        opening = detail == 0 and not self.board.stones
        for n, index in enumerate(indices):
            if n % 32 == 0:
                self.deadline.check()
            if not ((state.allowed >> index) & 1):
                continue
            point = self.points[index]
            if opening and (point[0] > point[1] or point[1] > self.board.size // 2):
                continue
            kernel = field.kernel(point)
            for weight, span, unit, prior, margin in tests:
                added = margin + weight * kernel
                if terminal:
                    value = 2 * field.wins(added) / field.count - 1
                else:
                    value = field.control(added, span) * unit - 1
                priority = value - prior
                if count == 1:
                    if best is None or priority > best.priority:
                        best = Choice(index, weight, value, priority)
                else:
                    ranking.append(Choice(index, weight, value, priority))
        if count == 1:
            return [best] if best is not None else []
        ranking.sort(key=lambda item: item.priority, reverse=True)
        selected = []
        radius = self.diversity if detail == 0 else max(1, self.board.size // 11) ** 2
        weight_step = max(2, left // turns // 3)
        for choice in ranking:
            if any(abs(choice.weight - other.weight) < weight_step and
                   distance_sq(self.points[choice.point], self.points[other.point]) < radius
                   for other in selected):
                continue
            selected.append(choice)
            if len(selected) == count:
                break
        if detail == 0:
            seen = {choice.weight for choice in selected}
            for choice in ranking:
                if choice.weight not in seen:
                    seen.add(choice.weight)
                    if all(abs(choice.weight - other.weight) >= weight_step for other in selected):
                        selected.append(choice)
        return selected

    def play(self, state, choice, field):
        actor = state.actor
        turns, reserve = list(state.turns), list(state.reserve)
        turns[actor] -= 1
        reserve[actor] -= choice.weight
        following = 1 - actor if turns[1 - actor] or not turns[actor] else actor
        margin = state.margin + choice.weight * field.kernel(self.points[choice.point])
        return State(margin if following == actor else -margin,
                     state.allowed & self.legal_mask(choice.point), tuple(turns), tuple(reserve),
                     following)

    def evaluate(self, state, field):
        margin = state.margin if state.actor == self.board.me else -state.margin
        if not any(state.turns):
            return 2 * field.wins(margin) / field.count - 1
        width = field.width(sum(state.reserve), self.board.total)
        return field.control(margin, width * field.ones) / (width * field.count) - 1

    def rollout(self, line, style, layout, horizon):
        field = self.fast_field
        state = self.fast_root
        for choice in line:
            state = self.play(state, choice, field)
        for _ in range(horizon):
            if not any(state.turns):
                break
            actor_style = 0 if state.actor == self.board.me else style
            choices = self.rank(state, 2, 1, field, actor_style, layout)
            if not choices:
                break
            state = self.play(state, choices[0], field)
        return self.evaluate(state, field)

    def minimax(self, state, depth, alpha, beta, width):
        self.deadline.check()
        if not depth or not any(state.turns):
            return self.evaluate(state, self.field)
        choices = self.rank(state, 1, 1 if depth == 1 else width)
        if not choices:
            return self.evaluate(state, self.field)
        maximize = state.actor == self.board.me
        if depth == 1 and sum(state.turns) == 1:
            return choices[0].value if maximize else -choices[0].value
        best = -2.0 if maximize else 2.0
        for choice in choices:
            value = self.minimax(self.play(state, choice, self.field), depth - 1, alpha, beta, width)
            if maximize:
                best = max(best, value)
                alpha = max(alpha, best)
            else:
                best = min(best, value)
                beta = min(beta, best)
            if alpha >= beta:
                break
        return best


def candidates(board, fallback, divisions):
    points = lattice(board.size, divisions)
    for point in board.occupied:
        points.extend(ring(point, board.spacing, 20))
    points.append(fallback)
    return [point for point in dict.fromkeys(points) if board.legal(point)]


def polish(point, value, step, evaluate, deadline):
    while step:
        for _ in range(2):
            start = point
            for dy in (-step, 0, step):
                for dx in (-step, 0, step):
                    if dx or dy:
                        deadline.check(0.025)
                        test = start[0] + dx, start[1] + dy
                        score = evaluate(test)
                        if score > value:
                            point, value = test, score
            if point == start:
                break
        step //= 2
    return point, value


class Endgame:
    def __init__(self, board, field, deadline):
        self.board = board
        self.field = field
        self.deadline = deadline
        pulls = [0] * board.players
        for player, point, weight in board.stones:
            deadline.check(0.035)
            pulls[player] += weight * field.pull(point)
        ours = pulls[board.me]
        rivals = [p for i, p in enumerate(pulls) if i != board.me]
        if len(rivals) > 1:
            # Per-lane maximum via guard-bit comparisons.
            rival = 0
            for pull in rivals:
                larger = ((field.guard + pull - rival) & field.guard) >> field.shift
                mask = larger * field.modulus
                rival = (rival & ~mask) | (pull & mask)
        else:
            rival = rivals[0] if rivals else 0
        self.margin = ours - rival
        self.weight = board.reserve[board.me]

    def attack(self, point):
        if not self.board.legal(point):
            return -1
        return self.field.wins(self.margin + self.weight * self.field.pull(point))

    def defend(self, point, options, cutoff=-float('inf')):
        board, field = self.board, self.field
        if not board.legal(point):
            return -float('inf'), None
        weight = board.reserve[1 - board.me]
        margin = -self.margin - self.weight * field.pull(point)

        def reply(other):
            if distance_sq(other, point) < board.radius_sq or not board.legal(other):
                return -1
            return field.wins(margin + weight * field.pull(other))

        strongest, best = None, -1
        for i, other in enumerate(options):
            if i % 24 == 0:
                self.deadline.check(0.035)
            if distance_sq(other, point) < board.radius_sq:
                continue
            score = field.wins(margin + weight * field.pull(other))
            if score > best:
                strongest, best = other, score
            if -best < cutoff:
                return -best, strongest
        for other in ring(point, board.spacing, 20):
            score = reply(other)
            if score > best:
                strongest, best = other, score
        if strongest is None:
            return 0, None
        strongest, best = polish(strongest, best, max(1, board.size // 89), reply, self.deadline)
        return -best, strongest


def last_move(board, points, fallback, deadline):
    best = fallback
    try:
        field = Field(board.size, 46, board.total, forecast=False)
        endgame = Endgame(board, field, deadline)
        ranking = []
        best_score = -1
        for point in points:
            deadline.check(0.040)
            score = endgame.attack(point)
            ranking.append((score, point))
            if score > best_score:
                best_score, best = score, point
        ranking.sort(reverse=True)
        finalists = []
        for score, point in ranking:
            if any(distance_sq(point, other) < (board.size // 27) ** 2 for other in finalists):
                continue
            point, score = polish(point, score, max(1, board.size // 58), endgame.attack, deadline)
            finalists.append(point)
            if score > best_score:
                best_score, best = score, point
            if len(finalists) >= 5:
                break
        for side in (88, 160):
            if deadline.left() < (0.30 if side == 88 else 0.65):
                break
            dense = Endgame(board, Field(board.size, side, board.total, forecast=False), deadline)
            best_score = dense.attack(best)
            for point in sorted(set(finalists + [best]), key=dense.attack, reverse=True)[:3]:
                point, score = polish(point, dense.attack(point), max(1, board.size // (side * 2)),
                                      dense.attack, deadline)
                if score > best_score:
                    best_score, best = score, point
    except SearchExpired:
        pass
    return best, board.reserve[board.me]


def last_duel(board, points, seeds, deadline):
    best = seeds[0]
    try:
        coarse = Endgame(board, Field(board.size, 36, board.total, forecast=False), deadline)
        best_score = -float('inf')
        ranked = []
        for point in seeds[:17]:
            score, reply = coarse.defend(point, points, best_score - 2)
            if score > best_score:
                best_score, best = score, point
            ranked.append((score, point, reply))
        if deadline.left() < 0.22:
            return best
        dense = Endgame(board, Field(board.size, 66, board.total, forecast=False), deadline)
        replies = [reply for _, _, reply in sorted(ranked, reverse=True)[:10] if reply is not None]
        # Search the strongest reply again on the denser field, prioritizing known counters.
        options = list(dict.fromkeys(replies + points[::3]))
        best_score = -float('inf')

        def defend(point):
            return dense.defend(point, options, best_score - 3)[0]

        for _, point, _ in sorted(ranked, reverse=True)[:7]:
            score = defend(point)
            if score > best_score:
                best_score, best = score, point
        best, _ = polish(best, best_score, max(1, board.size // 110), defend, deadline)
    except SearchExpired:
        pass
    return best


class Bot:
    def start(self):
        self.fields = None

    def move(self, msg):
        board = Board(msg)
        me = board.me
        turns, left = board.turns[me], board.reserve[me]
        if turns <= 0 or left <= 0:
            return 0, 0, 0
        deadline = Deadline(msg['clock'], turns)
        fallback = board.fallback()
        if fallback is None:
            return 0, 0, 0
        best = fallback, max(1, left // turns)
        if deadline.left() < 0.065 or board.size == 1:
            return *best[0], best[1]
        if turns == 1 and all(p == me or not min(t, w)
                              for p, (t, w) in enumerate(zip(board.turns, board.reserve))):
            point, weight = last_move(board, candidates(board, fallback, 26), fallback, deadline)
            return *point, weight
        try:
            if self.fields is None:
                medium_game = 5 < board.stones_each <= 12
                side, fast_side = (24, 14) if medium_game else (20, 11)
                self.fields = Field(board.size, side, board.total), Field(board.size, fast_side, board.total)
            field, fast = self.fields
            points = candidates(board, fallback, 28 if board.stones_each <= 12 else 25)
            if board.players == 2:
                best = self.pair(board, field, fast, points, deadline, best)
            else:
                endgame = Endgame(board, field, deadline)
                best_score = -1
                for point in points:
                    deadline.check()
                    score = field.wins(endgame.margin + best[1] * field.pull(point))
                    if score > best_score:
                        best_score, best = score, (point, best[1])
        except SearchExpired:
            pass
        return *best[0], best[1]

    def pair(self, board, field, fast, points, deadline, best):
        me = board.me
        search = PairSearch(board, field, fast, points, deadline)
        plies = sum(search.root.turns)
        # With few stones, spending the clock on deeper replies is more valuable.
        short_game = board.stones_each <= 5
        medium_game = 5 < board.stones_each <= 12
        root_width = 34 if short_game else 38 if medium_game else 26
        final_duel = search.root.turns == (1, 1)
        try:
            root = search.rank(search.root, 0, root_width)
            if not root:
                return best
            best = points[root[0].point], root[0].weight
            # Refine both position and allocation around the strongest opening candidates.
            if deadline.left() > 0.55 and not final_duel:
                nearby = []
                step = max(1, board.size // 76)
                for choice in root[:20 if short_game else 18 if medium_game else 14]:
                    x, y = points[choice.point]
                    nearby.extend((x + dx, y + dy) for dx, dy in
                                  ((step, 0), (-step, 0), (0, step), (0, -step)))
                refined = list(dict.fromkeys(points + [p for p in nearby if board.legal(p)]))
                refined_search = PairSearch(board, field, fast, refined, deadline)
                refined_root = refined_search.rank(refined_search.root, 0, root_width)
                if refined_root:
                    points, search, root = refined, refined_search, refined_root
                    best = points[root[0].point], root[0].weight

            lines = []
            best_estimate = -2.0
            for choice in root:
                if deadline.left() < 0.085:
                    break
                after = search.play(search.root, choice, field)
                reply_state = search.play(search.fast_root, choice, fast)
                replies = search.rank(reply_state, 1, 3, fast)
                if not replies:
                    continue
                sign = 1 if after.actor == me else -1
                estimate = sign * replies[0].value
                if not final_duel:
                    estimate = 0.22 * choice.value + 0.78 * estimate
                lines.append((estimate, choice, replies, after))
                if estimate > best_estimate:
                    best_estimate = estimate
                    best = points[choice.point], choice.weight
            if not lines:
                return best
            lines.sort(key=lambda line: line[0], reverse=True)

            if final_duel:
                seeds = [points[choice.point] for _, choice, _, _ in lines]
                return last_duel(board, points, seeds, deadline), best[1]
            if plies <= (6 if short_game else 5):
                best_score = -2.0
                for estimate, choice, _, after in lines[:17 if short_game else 13]:
                    deadline.check(0.045)
                    depth = plies - 1 if short_game else min(plies - 1, 3)
                    alpha = (best_score - 0.12 * estimate) / 0.88
                    terminal = search.minimax(after, depth, alpha, 2.0, 3 if short_game else 4)
                    score = 0.12 * estimate + 0.88 * terminal
                    if score > best_score:
                        best_score, best = score, (points[choice.point], choice.weight)
                return best

            forecasts = []
            best_score = -2.0
            shallow_width = 25 if short_game else 24 if medium_game else 18
            horizon = min(plies - 2, 16 if medium_game else 10)
            for estimate, choice, replies, _ in lines[:shallow_width]:
                deadline.check(0.045)
                first = search.rollout((choice, replies[0]), 0, 0, horizon)
                second = search.rollout((choice, replies[min(1, len(replies) - 1)]),
                                        2, 1, horizon)
                score = 0.27 * estimate + 0.73 * (0.64 * min(first, second) + 0.36 * max(first, second))
                forecasts.append((score, estimate, choice, replies))
                if score > best_score:
                    best_score, best = score, (points[choice.point], choice.weight)
            forecasts.sort(key=lambda line: line[0], reverse=True)
            best_score = -2.0
            deep_width = 12 if short_game else 10 if medium_game else 8
            for forecast, estimate, choice, replies in forecasts[:deep_width]:
                deadline.check(0.060)
                after = search.play(search.fast_root, choice, fast)
                immediate = search.rank(after, 1, 1, fast, immediate=True)
                counters = list(dict.fromkeys(replies[:2] + immediate))
                values = []
                for layout, reply in enumerate(counters):
                    state = search.play(after, reply, fast)
                    responses = search.rank(state, 2, 2, fast, layout=layout % 2)
                    horizon = min(plies - 3, 23 if medium_game else 15)
                    values.append(max(search.rollout((choice, reply, response), 1 + layout % 2,
                                                     layout % 2, horizon) for response in responses)
                                  if responses else search.evaluate(state, fast))
                terminal = 0.58 * min(values) + 0.42 * sum(values) / len(values)
                score = 0.26 * estimate + 0.19 * forecast + 0.55 * terminal
                if score > best_score:
                    best_score, best = score, (points[choice.point], choice.weight)
        except SearchExpired:
            pass
        return best


def main():
    bot = Bot()
    for line in sys.stdin:
        if not line.strip():
            continue
        msg = json.loads(line)
        if msg['type'] == 'start':
            bot.start()
            reply = {'ready': True}
        elif msg['type'] == 'move':
            x, y, weight = bot.move(msg)
            reply = {'x': x, 'y': y, 'weight': weight}
        else:
            reply = {}
        print(json.dumps(reply, separators=(',', ':')), flush=True)


if __name__ == '__main__':
    main()
