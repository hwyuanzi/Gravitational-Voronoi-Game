# Gravitational Voronoi

The architecture for the Heuristic Problem Solving game **Gravitational Voronoi** (Dennis
Shasha): a referee that runs every bot on one machine, a live dashboard in the browser, saved
replays and logs, and starter bots in **Python, C++ and Julia**. It is set up the same way as
the Evasion, Kelly and PegBand architectures.

Players send their bot folders to the architect. The architect drops them into `bots/` and runs
the tournament locally, so no bot talks over the network.

| | |
|---|---|
| **Runs on** | Mac, Linux and Windows |
| **Needs** | Python 3.8+, standard library only (nothing to `pip install`) |
| **Board** | 1000 × 1000 integer points; stones at least 66 apart |
| **Parameters** | stones per player, total weight per player (a parameter file) |
| **Clock** | 120 seconds of thinking time per bot per game; bots think one at a time |
| **A pairing** | 2 games: each bot moves first once. A bot's score is its territory over both games |
| **A melee** | k bots, k games: the seats rotate so every bot sits in every seat once |

**Contents:** [Quick start](#quick-start) · [Adding players' bots](#adding-players-bots) ·
[The rules](#the-rules-the-referee-enforces) · [The dashboard](#the-dashboard) ·
[Making a bot](#making-a-bot) · [The messages](#the-messages-between-referee-and-bot) ·
[Other ways to run](#other-ways-to-run)

---

## Quick start

```sh
cd gravitational_voronoi
python3 start.py
```

Open **http://127.0.0.1:8093** in your browser. Stop the server with **Ctrl+C**.
(On Windows, type `python` instead of `python3`.)

Press **Every pair** (or **Melee of every bot**), then **Start tournament**, and the sample bots
and contest entry play each other while you watch.

## Contest entry

The submission is [`bots/haowen/bot.py`](bots/haowen/bot.py). It runs on Python 3.8+ with
only the standard library. The referee discovers it as `haowen`. The original
[`bot.cpp`](bots/haowen/bot.cpp) is kept for comparison. Run a pairing and verify its
results from the repository root.

```sh
python3 start.py match haowen "Greedy (Python)" --config params/s10_w100.txt --no-browser
python3 start.py verify results/match/results.json
```

---

## Adding players' bots

Each player sends their bot as a folder holding a `bot.py`, `bot.cpp` or `bot.jl` (made from one
of the templates in `bots/templates/`). Unzip it into `bots/`:

```
bots/
  alice/bot.py
  bob/bot.cpp       (and build.py, if they changed how it compiles)
  carol/bot.jl
```

Press **Rescan** on the dashboard and they appear, named after their folders. C++ bots are
compiled before any clock starts; Julia bots need `julia` on the PATH.

To give a bot a nicer name or a different command, list it in `bots.local.json` (same format as
`bots.json`). Git tracks the samples, templates, and contest submission. It ignores
`bots.local.json` and folders for other players' bots.

---

## The rules the referee enforces

**Placing stones.** Each player has the same number of stones and the same total weight, from
the parameter file. Player 1 (red) places a stone, then player 2 (blue), and so on, until every
player has had one turn per stone, so the last stone is placed by the last player. A stone is
legal when:

* `x` and `y` are whole numbers from 0 to 999 (x left to right, y top to bottom),
* it is at least 66 from every stone already on the board, of any colour (exactly 66 is fine),
* its `weight` is a whole number, at least 1 and no more than the player's weight left.

A player should keep at least 1 weight for each stone still to place.

**An illegal stone skips the turn.** The stone is not placed and that turn is lost, but the game
goes on and the bot is asked again on its next turn. Each skip is written:

* to the referee notes on the dashboard and in the replay, in red,
* as a "turn skipped" banner on the board, showing exactly what the bot sent,
* to the match's `log.txt` and to the server console,
* as a count in the results table.

**The score.** Every point of the board belongs to the colour with the greatest pull:

    pull(c, p) = Σ over the stones s of colour c:  weight(s) / distance(s, p)²

A point holding a stone always belongs to that stone's colour (its pull there is infinite).
Where two or more colours tie for the greatest pull, the point belongs to nobody (grey on the
board). Pulls are floating point, so a tie means equal to within one part in 10⁹. A player's
score is the number of points it owns. `tests/test_game.py` checks the referee's scoring point
by point against exact fractions.

**Matches.** A pairing is two games with the same parameters, each bot moving first once, and a
bot's score is its territory over both games added together, as the spec says. A melee does the
same with more bots: one game per bot, the seating rotating so every bot sits in every seat
exactly once (including the last seat, which places the last stone). The highest total wins.

**Clocks and failure.** Each bot has 120 seconds of thinking time for a whole game (set on the
dashboard or with `--clock`), counted only while the referee waits for its answer. The
referee's own scoring time is never charged to anyone. A bot that runs out of time, crashes, or
breaks the protocol (a reply that is not one line of JSON) **forfeits**: it places nothing more,
its remaining turns are skipped, and it keeps the territory its stones already hold. The other
players play on, and the result is marked forfeit so a win that way is never mistaken for a win
on play. A stone that arrives after the clock runs out is not placed.

A C++ bot whose build fails forfeits with the compiler's message. The referee deletes the
binary left by the last good build, so nobody is scored on code that no longer compiles.

---

## The dashboard

1. **Bots** are read from `bots.json`, plus any folder dropped into `bots/`. Press **Rescan**
   after adding one.
2. **Parameters** are read from `params/`: each file holds the stones per player and the total
   weight per player. Type new ones into the page, or drop a file in and press **Rescan**.
3. **Matches:** **Add pairing** for two bots, **+ bot** on a row to turn it into a melee,
   **Every pair** for a round robin, or **Melee of every bot**. Choose a parameter file for each.
   Then **Start tournament**.

While a tournament runs, **Playing now** shows the board live: each colour's territory in a
light shade, the stones as darker dots (bigger means heavier), every player's points, share of
the board, weight left and clock, and whose turn it is. Tick **show the no-go zone** to see the
66-point circle around each stone. Hover over the board to read the coordinates and owner of
any point. The **watching pace** lingers a few hundred milliseconds per move so a game is
watchable; it never counts against the bots' clocks and can be changed while a game runs.

Finished matches go into the results table with each game's scores in seat order, the totals,
the winner and margin, and links to the **replay** and the **log**. Replays are self-contained
HTML files under `results/`: step through one with the arrow keys or press space to play.
Results survive a restart, and **Clear results** keeps a backup.

---

## Making a bot

Copy a template out of `bots/templates/` into `bots/<your-name>/`:

```sh
cp -r bots/templates/python bots/my-bot
```

Edit the part marked `EDIT ONLY THIS SECTION`: one function that returns `(x, y, weight)` for
your next stone. The templates place a random legal stone with the weight split evenly. Working
examples are in `bots/samples/`:

| Bot | What it does |
|---|---|
| **Greedy** (Python) | Tries a lattice of legal points and keeps the one that would own the most of a coarse sample of the board |
| **Spread** (Python) | Puts each stone as far as possible from every stone already placed |
| **Random** (Python, C++) | The template unchanged: a random legal point. A punching bag |

A bot in another language just needs to read a line of JSON and write a line of JSON; list it
in `bots.json` with the command that runs it.

Flush after every reply. C++ (`std::flush`), Julia (`flush(stdout)`) and buffered Python will
otherwise hold the answer until the buffer fills, and the referee times you out. Print debugging
output to standard error only: a stray `print` on standard output lands in the middle of the
protocol and forfeits your game.

---

## The messages between referee and bot

One JSON object per line, on standard input and standard output. Anything on standard error is
ignored.

**Start** (not on your clock, 30 seconds to reply):

```json
{"type":"start","protocol_version":1,"you":1,"players":2,"color":"red","grid":1000,"min_dist":66,"stones":10,"weight":100,"clock":120}
```
reply `{"ready": true}`

**Your move:** reply `{"x": 500, "y": 500, "weight": 10}`. The message carries the whole
position, so a bot never has to remember anything:

| field | meaning |
|---|---|
| `you`, `players`, `color` | your seat (1 moves first), how many players, your colour |
| `grid`, `min_dist` | the board size and the minimum distance between stones |
| `stones_each`, `weight_each` | the parameters of this game |
| `turn` | which of your stones this is, 1 .. `stones_each` |
| `stones` | every stone so far, in order, as `[player, x, y, weight]` (player is 1-based) |
| `stones_left`, `weight_left` | turns and weight each player has left (index 0 is player 1) |
| `scores` | points each player owns right now |
| `clock` | your thinking time left, in seconds |

Not every language's JSON encoder keeps integers looking like integers, so `500`, `500.0` and
`"500"` are all read as 500. A real fraction such as `500.5` is not a whole number, so that
stone is refused and the turn skipped.

---

## Other ways to run

```sh
python3 start.py match "Greedy (Python)" "Spread (Python)"           # one pairing, opens the replay
python3 start.py match "A" "B" "C" --config params/s20_w1000.txt     # a three-bot melee
python3 start.py tournament                                          # every pair, no browser
python3 start.py tournament --melee                                  # one melee of every bot
python3 start.py params --stones 10 --weight 100 --out params/my_game.txt
python3 -m unittest discover -s tests -t .                           # the rules and the referee
```

`--help` after any command lists its options. The command-line runs print every skip and
forfeit as it happens, and write `results.json`, `replay.html` and `log.txt` for each match.

## Checking that it plays by the rules

Two independent checks:

```sh
python3 -m unittest discover -s tests -t .              # 27 tests, a few seconds
python3 start.py verify results/match/results.json      # audit any finished match
```

The **tests** check each rule on its own: distance exactly 66 allowed and 65.6 refused, stones
off the board, weights of 0, -50, 2.5 and more than what is left, turn order, the seat rotation,
and the territory point by point against exact fractions (two players and a melee). They also
play real games against bots that misbehave: illegal every turn, crashing, timing out, printing
garbage, never saying ready.

**verify** re-reads a finished match from its `results.json` without using the referee's code
and prints PASS or FAIL for ten rules from the spec: seat rotation, turn order, board bounds,
the 66 distance, weights and budgets, that every skipped turn really was illegal, forfeits,
clocks, a full point-by-point recount of the 1000 × 1000 territory, and the totals and winner.
Run it on any tournament match, e.g. `results/tournament/<date>/01/results.json`.

## What is where

```
start.py              every command
referee/engine.py     the rules: legal stones, the pull, who owns every point, the score
referee/runner.py     runs the bot processes, the clocks, one game and one match
referee/server.py     the dashboard
referee/tournament.py tournaments from the command line
referee/match.py      one match from the command line
referee/verify.py     checks a finished match against every rule, without the referee's code
dashboard/            the dashboard page, and the board view (also used for replays)
bots/templates/       starter bots in Python, C++ and Julia
bots/samples/         working bots to play against
params/               parameter files
tests/                the scoring checked against exact arithmetic, and misbehaving bots
```

The board size (1000), minimum distance (66) and tie tolerance are named constants at the top
of `referee/engine.py`, and are sent to the bots in every message, so nothing is a magic number.
