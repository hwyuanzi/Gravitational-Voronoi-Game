// Gravitational Voronoi, protocol version 1. Build with c++ -std=c++17 -O2.
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#if defined(__aarch64__) && defined(__ARM_NEON)
#include <arm_neon.h>
#elif defined(__SSE2__)
#include <xmmintrin.h>
#endif

namespace {

using Weight = long long;
using Clock = std::chrono::steady_clock;
constexpr double tieTolerance = 1.0e-9;
constexpr double pi = 3.141592653589793;

struct Point {
    int x;
    int y;
};

struct Move {
    Point at;
    Weight weight;
};

struct Stone {
    int player;
    Move move;
};

int distanceSquared(Point a, Point b) {
    const int dx = a.x - b.x;
    const int dy = a.y - b.y;
    return dx * dx + dy * dy;
}

const char* jsonValue(const std::string& line, const char* key) {
    const auto name = line.find(std::string("\"") + key + "\"");
    if (name == std::string::npos) return "";
    const auto colon = line.find(':', name);
    if (colon == std::string::npos) return "";
    const auto value = line.find_first_not_of(" \t\r\n", colon + 1);
    return value == std::string::npos ? "" : line.c_str() + value;
}

std::vector<Weight> jsonIntegers(const std::string& line, const char* key) {
    const char* p = jsonValue(line, key);
    std::vector<Weight> numbers;
    int depth = 0;
    if (*p != '[') return numbers;
    do {
        if (*p == '[') ++depth;
        if (*p == ']') --depth;
        char* end;
        if ((*p == '-' && p[1] >= '0' && p[1] <= '9') || (*p >= '0' && *p <= '9')) {
            numbers.push_back(std::strtoll(p, &end, 10));
            p = end;
        } else {
            ++p;
        }
    } while (*p && depth > 0);
    return numbers;
}

int jsonInt(const std::string& line, const char* key) {
    return static_cast<int>(std::strtol(jsonValue(line, key), nullptr, 10));
}

struct Board {
    explicit Board(const std::string& line)
        : size(jsonInt(line, "grid")), spacing(jsonInt(line, "min_dist")),
          me(jsonInt(line, "you") - 1), players(jsonInt(line, "players")),
          totalWeight(std::strtoll(jsonValue(line, "weight_each"), nullptr, 10)),
          timeLeft(std::strtod(jsonValue(line, "clock"), nullptr)),
          turns(jsonIntegers(line, "stones_left")), reserve(jsonIntegers(line, "weight_left")) {
        const auto history = jsonIntegers(line, "stones");
        for (std::size_t i = 0; i + 3 < history.size(); i += 4) {
            stones.push_back({static_cast<int>(history[i]) - 1,
                              {{static_cast<int>(history[i + 1]),
                                static_cast<int>(history[i + 2])}, history[i + 3]}});
        }
    }

    bool legal(Point p) const {
        if (p.x < 0 || p.x >= size || p.y < 0 || p.y >= size) return false;
        for (const Stone& s : stones) {
            if (distanceSquared(p, s.move.at) < spacing * spacing) return false;
        }
        return true;
    }

    Move fallback() const {
        const Weight w = std::max<Weight>(1, reserve[me] / turns[me]);
        if (stones.empty()) return {{size / 2, size / 2}, w};

        Point best{-1, -1};
        int clearance = spacing * spacing - 1;
        const int step = std::max(1, (size - 1) / 20);
        for (int y = 0; y < size; y += step) {
            for (int x = 0; x < size; x += step) {
                int nearest = std::numeric_limits<int>::max();
                for (const Stone& s : stones) {
                    nearest = std::min(nearest, distanceSquared({x, y}, s.move.at));
                    if (nearest <= clearance) break;
                }
                if (nearest > clearance) {
                    clearance = nearest;
                    best = {x, y};
                }
            }
        }
        if (best.x >= 0) return {best, w};

        // Skip over the rest of a forbidden disk on this scanline.
        const int radiusSquared = spacing * spacing;
        for (int y = 0; y < size; ++y) {
            for (int x = 0; x < size;) {
                int next = x;
                for (const Stone& s : stones) {
                    if (distanceSquared({x, y}, s.move.at) >= radiusSquared) continue;
                    const int dy = y - s.move.at.y;
                    const int radius = static_cast<int>(std::sqrt(radiusSquared - 1 - dy * dy));
                    next = std::max(next, s.move.at.x + radius + 1);
                }
                if (next == x) return {{x, y}, w};
                x = next;
            }
        }
        return {{0, 0}, 0};  // The board is packed. No legal move exists.
    }

    int size;
    int spacing;
    int me;
    int players;
    Weight totalWeight;
    double timeLeft;
    std::vector<Weight> turns;
    std::vector<Weight> reserve;
    std::vector<Stone> stones;
};

class Deadline {
public:
    explicit Deadline(const Board& board) {
        const double clock = board.timeLeft;
        const Weight left = board.turns[board.me];
        const double fraction = left <= 2 ? 1.45 : 0.94;
        const double share = fraction * clock / static_cast<double>(left);
        const double slack = std::clamp(0.06 * clock, 0.075, 0.50);
        const double budget = std::max(0.0, std::min({clock - slack, share, 12.0}));
        until_ = Clock::now() + std::chrono::duration_cast<Clock::duration>(
                                    std::chrono::duration<double>(budget));
    }

    bool expired(double guard = 0.0) const {
        return std::chrono::duration<double>(until_ - Clock::now()).count() <= guard;
    }

private:
    Clock::time_point until_;
};

std::uint32_t mix(std::uint32_t x) {
    x ^= x >> 16;
    x *= 0x7feb352dU;
    x ^= x >> 15;
    x *= 0x846ca68bU;
    return x ^ (x >> 16);
}

// Stratified samples with reflected jitter avoid a lattice bias toward any corner.
std::vector<Point> samplePoints(int size, int side) {
    std::vector<Point> points;
    points.reserve(side * side);
    for (int row = 0; row < side; ++row) {
        const int a = std::min(row, side - 1 - row);
        for (int col = 0; col < side; ++col) {
            const int b = std::min(col, side - 1 - col);
            const auto jitter = [side](int u, int v) {
                return 0.13 + 0.74 * (mix(1U + u + 113U * v + 1093U * side) % 1024U) / 1024.0;
            };
            int x = static_cast<int>((b + jitter(b, a)) * size / side);
            int y = static_cast<int>((a + jitter(a, b)) * size / side);
            if (col != b) x = size - 1 - x;
            if (row != a) y = size - 1 - y;
            points.push_back({x, y});
        }
    }
    return points;
}

std::vector<Point> fullBoard(int size) {
    std::vector<Point> points;
    points.reserve(size * size);
    for (int y = 0; y < size; ++y) {
        for (int x = 0; x < size; ++x) points.push_back({x, y});
    }
    return points;
}

std::vector<Point> ring(Point center, int radius, int directions) {
    std::vector<Point> points;
    points.reserve(directions);
    for (int i = 0; i < directions; ++i) {
        const double angle = i * 2.0 * pi / directions;
        Point p{center.x + static_cast<int>(std::lround(radius * std::cos(angle))),
                center.y + static_cast<int>(std::lround(radius * std::sin(angle)))};
        // Rounding toward the center can place a point just inside the forbidden disk.
        if (distanceSquared(p, center) < radius * radius) {
            if (std::abs(p.x - center.x) >= std::abs(p.y - center.y)) {
                p.x += p.x > center.x ? 1 : -1;
            } else {
                p.y += p.y > center.y ? 1 : -1;
            }
        }
        points.push_back(p);
    }
    return points;
}

// Estimate the contribution of unspent mass from potential future sites. The softened
// kernel spreads it over the board and accounts for the smaller pull near the edges.
float futurePull(Point p, int size) {
    constexpr int side = 12;
    const float x = (static_cast<float>(p.x) + 0.5f) / size;
    const float y = (static_cast<float>(p.y) + 0.5f) / size;
    float total = 0;
    for (int row = 0; row < side; ++row) {
        const float dy = y - (row + 0.5f) / side;
        for (int col = 0; col < side; ++col) {
            const float dx = x - (col + 0.5f) / side;
            total += 1.0f / (dx * dx + dy * dy + 0.016f);
        }
    }
    return total / (side * side);
}

class SearchSpace {
public:
    SearchSpace(const Board& board, Point fallback)
        : points(samplePoints(board.size, std::min(40, board.size))), gridArea(board.size * board.size),
          spacingSquared(board.spacing * board.spacing), forecast(points.size()), seen_(gridArea) {
        constexpr int divisions = 40;
        for (int row = 0; row <= divisions; ++row) {
            for (int col = 0; col <= divisions; ++col) {
                const Point p{(col * (board.size - 1) + divisions / 2) / divisions,
                              (row * (board.size - 1) + divisions / 2) / divisions};
                constexpr int lattice[2][2] = {{0, -1}, {2, 1}};
                add(board, p, lattice[row % 2][col % 2]);
            }
        }
        for (const Stone& s : board.stones) {
            const auto nearby = ring(s.move.at, board.spacing, 24);
            for (std::size_t i = 0; i < nearby.size(); ++i) {
                add(board, nearby[i], i % 6 == 0 ? 0 : i % 6 == 2 ? 1 : i % 6 == 4 ? 2 : -1);
            }
        }
        add(board, fallback, 0);

        for (std::size_t i = 0; i < points.size(); ++i) {
            forecast[i] = futurePull(points[i], board.size);
        }
        cacheKernels();
    }

    void densify(const Board& board, const std::vector<Point>& centers) {
        const int step = std::max(1, board.size / 85);
        for (Point center : centers) {
            for (int dy : {-step, 0, step}) {
                for (int dx : {-step, 0, step}) {
                    add(board, {center.x + dx, center.y + dy}, -1);
                }
            }
        }
        cacheKernels();
    }

    const float* kernel(int id) const {
        return kernels.data() + id * points.size();
    }

    float pull(Point at, Point p) const {
        const int d2 = distanceSquared(at, p);
        return d2 ? static_cast<float>(gridArea) / static_cast<float>(d2) : 1.0e9f;
    }

    std::vector<Point> points;
    std::vector<Point> locations;
    std::array<std::vector<int>, 3> rollout;
    int gridArea;
    int spacingSquared;
    std::vector<float> forecast;

private:
    void add(const Board& board, Point p, int lattice) {
        if (!board.legal(p) || seen_[p.y * board.size + p.x]) return;
        seen_[p.y * board.size + p.x] = 1;
        if (lattice >= 0) rollout[lattice].push_back(static_cast<int>(locations.size()));
        locations.push_back(p);
    }

    void cacheKernels() {
        const std::size_t first = kernels.size() / points.size();
        kernels.resize(locations.size() * points.size());
        for (std::size_t id = first; id < locations.size(); ++id) {
            for (std::size_t i = 0; i < points.size(); ++i) {
                kernels[id * points.size() + i] = pull(locations[id], points[i]) - forecast[i];
            }
        }
    }

    std::vector<unsigned char> seen_;
    std::vector<float> kernels;
};

// Clipped signed margins give partial credit near a boundary on the coarse grid.
// Later in the search the width shrinks toward a count of controlled points.
float control(const float* margin, const float* kernel, int count, float mass, float width) {
    int i = 0;
    float sum = 0;
#if defined(__aarch64__) && defined(__ARM_NEON)
    const auto upper = vdupq_n_f32(width);
    const auto lower = vdupq_n_f32(-width);
    auto first = vdupq_n_f32(0);
    auto second = first;
    for (; i + 8 <= count; i += 8) {
        const auto a = vfmaq_n_f32(vld1q_f32(margin + i), vld1q_f32(kernel + i), mass);
        const auto b = vfmaq_n_f32(vld1q_f32(margin + i + 4), vld1q_f32(kernel + i + 4), mass);
        first = vaddq_f32(first, vminq_f32(upper, vmaxq_f32(lower, a)));
        second = vaddq_f32(second, vminq_f32(upper, vmaxq_f32(lower, b)));
    }
    sum = vaddvq_f32(vaddq_f32(first, second));
#elif defined(__SSE2__)
    const auto upper = _mm_set1_ps(width);
    const auto lower = _mm_set1_ps(-width);
    const auto weight = _mm_set1_ps(mass);
    auto first = _mm_setzero_ps();
    auto second = first;
    for (; i + 8 <= count; i += 8) {
        const auto a = _mm_add_ps(_mm_loadu_ps(margin + i),
                                  _mm_mul_ps(weight, _mm_loadu_ps(kernel + i)));
        const auto b = _mm_add_ps(_mm_loadu_ps(margin + i + 4),
                                  _mm_mul_ps(weight, _mm_loadu_ps(kernel + i + 4)));
        first = _mm_add_ps(first, _mm_min_ps(upper, _mm_max_ps(lower, a)));
        second = _mm_add_ps(second, _mm_min_ps(upper, _mm_max_ps(lower, b)));
    }
    float partial[4];
    _mm_storeu_ps(partial, _mm_add_ps(first, second));
    sum = partial[0] + partial[1] + partial[2] + partial[3];
#endif
    for (; i < count; ++i) {
        sum += std::clamp(margin[i] + mass * kernel[i], -width, width);
    }
    return sum / (width * count);
}

struct State {
    // The margin is oriented toward the player whose turn it is.
    std::vector<float> margin;
    std::vector<unsigned char> allowed;
    std::array<Weight, 2> turns;
    std::array<Weight, 2> reserve;
    int actor;
};

struct Choice {
    int id = -1;
    Weight weight = 0;
    float value = -2;
    float priority = -2;
};

enum class SearchDepth { Root, Reply, Rollout };

Weight weightLimit(Weight left, Weight turns) {
    return std::max<Weight>(1, left - (turns - 1));
}

std::vector<Weight> weightChoices(Weight left, Weight turns, SearchDepth detail, int style = 0) {
    const Weight maxWeight = weightLimit(left, turns);
    if (turns <= 1 || maxWeight <= 1) return {maxWeight};

    std::vector<Weight> choices;
    const auto add = [&](double proposed) {
        const double limit = static_cast<double>(maxWeight);
        const Weight weight = proposed >= limit ? maxWeight :
                              std::max<Weight>(1, static_cast<Weight>(std::llround(proposed)));
        choices.push_back(weight);
    };
    const double average = static_cast<double>(left) / static_cast<double>(turns);
    if (turns == 2 && detail == SearchDepth::Root && maxWeight <= 24) {
        for (Weight w = 1; w <= maxWeight; ++w) choices.push_back(w);
    } else if (turns == 2) {
        if (detail == SearchDepth::Root) {
            for (double fraction : {0.0, 0.10, 0.20, 0.30, 0.40, 0.50,
                                    0.60, 0.70, 0.80, 0.90, 1.0}) add(fraction * left);
        } else if (detail == SearchDepth::Reply) {
            for (double fraction : {0.0, 0.16, 0.33, 0.50, 0.67, 0.84, 1.0}) add(fraction * left);
        } else {
            for (double fraction : {0.26, 0.51, 0.76}) add(fraction * left);
        }
    } else if (detail == SearchDepth::Root) {
        for (double scale : {0.42, 0.62, 0.82, 1.0, 1.22, 1.48, 1.80, 2.15}) add(scale * average);
        if (turns <= 4) {
            add(1);
            add(0.68 * left);
        }
    } else if (detail == SearchDepth::Reply) {
        for (double scale : {0.48, 0.79, 1.0, 1.30, 1.72}) add(scale * average);
        if (turns <= 3) add(0.69 * left);
    } else if (style == 1) {
        for (double scale : {0.53, 0.78, 1.06}) add(scale * average);
    } else if (style == 2) {
        for (double scale : {0.92, 1.28, 1.72}) add(scale * average);
    } else {
        for (double scale : {0.66, 0.95, 1.26}) add(scale * average);
    }
    std::sort(choices.begin(), choices.end());
    choices.erase(std::unique(choices.begin(), choices.end()), choices.end());
    return choices;
}

class PairSearch {
public:
    PairSearch(const Board& board, const SearchSpace& space, const Deadline& deadline)
        : board_(board), space_(space), deadline_(deadline),
          unit_(1.0 / static_cast<double>(board.totalWeight)) {}

    State root() const {
        State state;
        state.actor = board_.me;
        for (int p = 0; p < 2; ++p) {
            state.turns[p] = std::min(board_.turns[p], board_.reserve[p]);
            state.reserve[p] = state.turns[p] > 0 ? board_.reserve[p] : 0;
        }
        state.allowed.assign(space_.locations.size(), 1);
        state.margin.resize(space_.points.size());
        const float spare = mass(state.reserve[state.actor]) - mass(state.reserve[1 - state.actor]);
        for (std::size_t i = 0; i < state.margin.size(); ++i) {
            float value = spare * space_.forecast[i];
            for (const Stone& s : board_.stones) {
                const float sign = s.player == state.actor ? 1.0f : -1.0f;
                value += sign * mass(s.move.weight) * space_.pull(s.move.at, space_.points[i]);
            }
            state.margin[i] = value;
        }
        return state;
    }

    std::vector<Choice> rank(const State& state, SearchDepth detail, int count,
                             int style = 0, int lattice = 0) const {
        const auto weights = weightChoices(state.reserve[state.actor], state.turns[state.actor],
                                           detail, style);
        const int samples = static_cast<int>(state.margin.size());
        const bool small = detail == SearchDepth::Rollout;
        const auto& subset = space_.rollout[lattice];
        const auto breadth = small ? subset.size() : space_.locations.size();
        const bool opening = detail == SearchDepth::Root && board_.stones.empty();
        struct TestWeight {
            Weight weight;
            float mass;
            float width;
            float prior;
        };
        std::vector<TestWeight> tests;
        for (Weight w : weights) {
            tests.push_back({w, mass(w), softness(state.reserve[state.actor] - w,
                                                 state.reserve[1 - state.actor]),
                             weightPrior(state, w, style)});
        }
        std::vector<Choice> ranking;
        if (count > 1) ranking.reserve(breadth * weights.size());
        Choice best;
        for (std::size_t k = 0; k < breadth; ++k) {
            if (k % 64 == 0 && deadline_.expired(0.003)) break;
            const int id = small ? subset[k] : static_cast<int>(k);
            if (!state.allowed[id]) continue;
            const Point p = space_.locations[id];
            if (opening && (p.x > p.y || p.y > board_.size / 2)) continue;
            const float* kernel = space_.kernel(id);
            for (const auto& test : tests) {
                const float value = control(state.margin.data(), kernel, samples, test.mass, test.width);
                const Choice choice{id, test.weight, value, value - test.prior};
                if (count > 1) ranking.push_back(choice);
                if (choice.priority > best.priority) best = choice;
            }
        }
        if (count == 1) return best.id >= 0 ? std::vector<Choice>{best} : std::vector<Choice>{};
        std::stable_sort(ranking.begin(), ranking.end(),
                         [](const Choice& a, const Choice& b) { return a.priority > b.priority; });
        return diversify(ranking, state, count, detail);
    }

    State play(const State& state, const Choice& choice) const {
        State child = state;
        child.reserve[state.actor] -= choice.weight;
        --child.turns[state.actor];
        child.actor = 1 - state.actor;
        if (!child.turns[child.actor] && child.turns[state.actor]) child.actor = state.actor;
        const float sign = child.actor == state.actor ? 1.0f : -1.0f;
        const float weight = mass(choice.weight);
        const float* kernel = space_.kernel(choice.id);
        for (std::size_t i = 0; i < child.margin.size(); ++i) {
            child.margin[i] = sign * (state.margin[i] + weight * kernel[i]);
        }
        const Point at = space_.locations[choice.id];
        for (std::size_t i = 0; i < child.allowed.size(); ++i) {
            child.allowed[i] &= distanceSquared(at, space_.locations[i]) >= space_.spacingSquared;
        }
        return child;
    }

    float evaluate(const State& state) const {
        const float width = softness(state.reserve[0], state.reserve[1]);
        float sum = 0.0f;
        for (float value : state.margin) sum += std::clamp(value, -width, width);
        return sum / (width * state.margin.size());
    }

    float rollout(State state, int perspective, int style, int lattice) const {
        while ((state.turns[0] || state.turns[1]) && !deadline_.expired(0.010)) {
            const int actorStyle = state.actor == perspective ? 0 : style;
            auto choices = rank(state, SearchDepth::Rollout, 1, actorStyle, lattice);
            if (choices.empty() && !deadline_.expired(0.010)) {
                choices = rank(state, SearchDepth::Reply, 1);
            }
            if (choices.empty()) break;
            state = play(state, choices.front());
        }
        return evaluate(state) * (state.actor == perspective ? 1.0f : -1.0f);
    }

    float minimax(const State& state, int perspective, int depth, float alpha, float beta) const {
        const float sign = state.actor == perspective ? 1.0f : -1.0f;
        if (!depth || !(state.turns[0] || state.turns[1]) || deadline_.expired(0.012)) {
            return sign * evaluate(state);
        }
        const int width = depth == 1 ? 1 : depth == 2 ? 6 : 8;
        const auto choices = rank(state, SearchDepth::Reply, width);
        if (choices.empty()) return sign * evaluate(state);
        if (depth == 1) return sign * choices.front().value;
        float best = -2.0f * sign;
        for (const Choice& choice : choices) {
            if (deadline_.expired(0.010) && best != -2.0f * sign) break;
            const float score = minimax(play(state, choice), perspective, depth - 1, alpha, beta);
            if (sign > 0) {
                best = std::max(best, score);
                alpha = std::max(alpha, best);
            } else {
                best = std::min(best, score);
                beta = std::min(beta, best);
            }
            if (alpha >= beta) break;
        }
        return best;
    }

    Choice immediateReply(const State& state) const {
        // A reactive opponent may optimize the current territory with a fixed
        // allocation. Include that reply even if it ranks lower after projection.
        const int actor = state.actor;
        const auto reserve = state.reserve[actor];
        const Weight turns = state.turns[actor];
        const Weight weight = reserve / turns + (reserve % turns >= (turns + 1) / 2);
        const float used = mass(weight);
        const float projected = mass(reserve - weight) - mass(state.reserve[1 - actor]);
        auto margin = state.margin;
        for (std::size_t i = 0; i < margin.size(); ++i) margin[i] -= projected * space_.forecast[i];
        Choice best;
        for (std::size_t id = 0; id < state.allowed.size(); ++id) {
            if (id % 64 == 0 && deadline_.expired(0.003)) break;
            if (!state.allowed[id]) continue;
            const float value = control(margin.data(), space_.kernel(static_cast<int>(id)),
                                        static_cast<int>(margin.size()), used, 0.12f);
            if (value > best.priority) {
                best = {static_cast<int>(id), weight, 0.0f, value};
            }
        }
        return best;
    }

    Move move(const Choice& choice) const {
        return {space_.locations[choice.id], choice.weight};
    }

private:
    float mass(Weight weight) const {
        return static_cast<float>(weight * unit_);
    }

    float softness(Weight first, Weight second) const {
        return 0.11f + 1.75f * (mass(first) + mass(second));
    }

    float weightPrior(const State& state, Weight weight, int style) const {
        const Weight turns = state.turns[state.actor];
        if (turns <= 2) return 0;
        const double average = static_cast<double>(state.reserve[state.actor]) / turns;
        const double target = style == 1 ? 0.72 : style == 2 ? 1.18 : 0.92;
        const double deviation = std::min(2.0, std::abs(weight / average - target));
        return static_cast<float>(0.016 * deviation * deviation);
    }

    std::vector<Choice> diversify(const std::vector<Choice>& ranking, const State& state,
                                  int count, SearchDepth detail) const {
        std::vector<Choice> selected;
        const int radius = detail == SearchDepth::Root ? board_.size / 20 : board_.size / 12;
        const Weight weightStep = std::max<Weight>(2, state.reserve[state.actor] /
                                                    state.turns[state.actor] / 3);
        for (const Choice& choice : ranking) {
            const bool similar = std::any_of(selected.begin(), selected.end(), [&](const Choice& s) {
                return distanceSquared(space_.locations[choice.id], space_.locations[s.id]) <
                       radius * radius && std::abs(choice.weight - s.weight) < weightStep;
            });
            if (similar) continue;
            selected.push_back(choice);
            if (static_cast<int>(selected.size()) == count) break;
        }
        if (detail == SearchDepth::Root) {
            // Keep some options with different commitments, even when the coarse
            // one-ply evaluation puts all of them below the main beam.
            std::vector<Weight> represented;
            for (const Choice& choice : ranking) {
                if (std::find(represented.begin(), represented.end(), choice.weight) != represented.end()) {
                    continue;
                }
                represented.push_back(choice.weight);
                const bool covered = std::any_of(selected.begin(), selected.end(), [&](const Choice& s) {
                    return std::abs(choice.weight - s.weight) < weightStep;
                });
                if (!covered) selected.push_back(choice);
            }
        }
        return selected;
    }

    const Board& board_;
    const SearchSpace& space_;
    const Deadline& deadline_;
    double unit_;
};

struct Target {
    double x;
    double y;
    double needed;
};

// Precompute the remaining pull needed to win each point. Scoring a hypothetical
// stone then only needs a multiplication and comparison per contested point.
struct Territory {
    std::vector<Target> targets;
    int owned = 0;

    int score(Point p, Weight w) const {
        int captured = 0;
        for (const Target& target : targets) {
            const double dx = target.x - p.x;
            const double dy = target.y - p.y;
            captured += (dx * dx + dy * dy) * target.needed < static_cast<double>(w);
        }
        return owned + captured;
    }
};

class PullMap {
public:
    PullMap(const Board& board, std::vector<Point> samples, const Deadline* deadline = nullptr)
        : points_(std::move(samples)), ours_(points_.size()), theirs_(points_.size()),
          occupied_(points_.size(), -1) {
        std::vector<double> other(points_.size());
        for (int p = 0; p < board.players; ++p) {
            auto& pull = p == board.me ? ours_ : other;
            std::fill(pull.begin(), pull.end(), 0.0);
            for (const Stone& s : board.stones) {
                if (s.player != p) continue;
                if (deadline && deadline->expired(0.035)) {
                    complete_ = false;
                    return;
                }
                for (std::size_t i = 0; i < points_.size(); ++i) {
                    const int d2 = distanceSquared(points_[i], s.move.at);
                    if (d2) {
                        pull[i] += static_cast<double>(s.move.weight) / d2;
                    } else {
                        occupied_[i] = p == board.me ? 0 : 1;
                    }
                }
            }
            if (p != board.me) {
                for (std::size_t i = 0; i < points_.size(); ++i) {
                    theirs_[i] = std::max(theirs_[i], other[i]);
                }
            }
        }
    }

    bool complete() const { return complete_; }

    Territory territory(bool ours, const Move* additional = nullptr) const {
        Territory territory;
        territory.targets.reserve(points_.size() / 2);
        for (std::size_t i = 0; i < points_.size(); ++i) {
            int occupied = occupied_[i];
            double ourPull = ours_[i];
            if (additional) {
                const int d2 = distanceSquared(points_[i], additional->at);
                if (d2) {
                    ourPull += static_cast<double>(additional->weight) / d2;
                } else {
                    occupied = 0;
                }
            }
            const int side = ours ? 0 : 1;
            if (occupied >= 0) {
                territory.owned += occupied == side;
                continue;
            }
            const double mine = ours ? ourPull : theirs_[i];
            const double rival = ours ? theirs_[i] : ourPull;
            const double needed = rival * (1.0 + tieTolerance) - mine;
            if (needed <= 0) {
                ++territory.owned;
            } else {
                territory.targets.push_back({static_cast<double>(points_[i].x),
                                             static_cast<double>(points_[i].y), needed});
            }
        }
        return territory;
    }

private:
    std::vector<Point> points_;
    std::vector<double> ours_;
    std::vector<double> theirs_;
    std::vector<signed char> occupied_;
    bool complete_ = true;
};

template <class Evaluate>
int polish(Point& at, int initialStep, const Deadline& deadline, Evaluate evaluate) {
    int best = evaluate(at);
    for (int step = initialStep; step > 0 && !deadline.expired(0.006); step /= 2) {
        for (int repeat = 0; repeat < 2 && !deadline.expired(0.006); ++repeat) {
            const Point start = at;
            for (int dy : {-step, 0, step}) {
                for (int dx : {-step, 0, step}) {
                    if ((!dx && !dy) || deadline.expired(0.003)) continue;
                    const Point test{start.x + dx, start.y + dy};
                    const int score = evaluate(test);
                    if (score > best) {
                        at = test;
                        best = score;
                    }
                }
            }
            if (distanceSquared(start, at) == 0) break;
        }
    }
    return best;
}

Move lastMove(const Board& board, const SearchSpace& space, const Deadline& deadline, Move fallback) {
    const Weight weight = board.reserve[board.me];
    Move best{fallback.at, weight};
    const auto dense = PullMap(board, samplePoints(board.size, std::min(88, board.size))).territory(true);
    std::vector<std::pair<int, Point>> ranking;
    for (Point p : space.locations) {
        if (deadline.expired(0.012)) break;
        ranking.push_back({dense.score(p, weight), p});
    }
    std::stable_sort(ranking.begin(), ranking.end(),
                     [](const auto& a, const auto& b) { return a.first > b.first; });
    std::vector<Point> finalists;
    int denseBest = -1;
    for (const auto& entry : ranking) {
        if (finalists.size() >= 7 || deadline.expired(0.030)) break;
        Point p = entry.second;
        const int radius = std::max(1, board.size / 26);
        if (std::any_of(finalists.begin(), finalists.end(), [&](Point q) {
                return distanceSquared(p, q) < radius * radius;
            })) continue;
        const int value = polish(p, std::max(1, board.size / 64), deadline, [&](Point q) {
            return board.legal(q) ? dense.score(q, weight) : -1;
        });
        if (value > denseBest) {
            denseBest = value;
            best.at = p;
        }
        finalists.push_back(p);
    }
    if (denseBest < 0 && !ranking.empty()) best.at = ranking.front().second;
    if (finalists.empty() || deadline.expired(0.23)) return best;

    const PullMap full(board, fullBoard(board.size), &deadline);
    if (!full.complete() || deadline.expired(0.045)) return best;
    const auto exact = full.territory(true);
    int bestScore = exact.score(best.at, weight);
    for (Point p : finalists) {
        if (deadline.expired(0.03)) break;
        const int score = polish(p, std::max(1, board.size / 135), deadline, [&](Point q) {
            return board.legal(q) ? exact.score(q, weight) : -1;
        });
        if (score > bestScore) {
            bestScore = score;
            best.at = p;
        }
    }
    return best;
}

// When we move first in the final round, optimize against the opponent's forced
// weight. The reply is re-searched after every change to our own position.
Move lastDuel(const Board& board, const SearchSpace& space, const Deadline& deadline,
              const std::vector<Move>& seeds) {
    if (deadline.expired(0.10)) return seeds.front();
    const PullMap pulls(board, samplePoints(board.size, std::min(108, board.size)));
    const Weight ourWeight = board.reserve[board.me];
    const Weight replyWeight = board.reserve[1 - board.me];
    Move best = seeds.front();
    constexpr int incomplete = std::numeric_limits<int>::min();
    int bestScore = incomplete;
    const auto defend = [&](Point p, int cutoff, std::vector<Point>* strongest) {
        if (!board.legal(p)) return incomplete;
        const Move ours{p, ourWeight};
        const Territory rival = pulls.territory(false, &ours);
        int replyScore = -1;
        Point reply{0, 0};
        std::vector<std::pair<int, Point>> candidates;
        const auto consider = [&](Point q) {
            if (distanceSquared(p, q) < space.spacingSquared) return;
            const int score = rival.score(q, replyWeight);
            if (strongest) candidates.push_back({score, q});
            if (score > replyScore) {
                replyScore = score;
                reply = q;
            }
        };
        for (Point q : space.locations) {
            if (deadline.expired(0.004)) return incomplete;
            consider(q);
            if (-replyScore < cutoff) return -replyScore;
        }
        for (Point q : ring(p, board.spacing, 20)) {
            if (board.legal(q)) consider(q);
        }
        if (replyScore < 0) return 0;
        const auto score = [&](Point q) {
            if (!board.legal(q) || distanceSquared(p, q) < space.spacingSquared) return -1;
            return rival.score(q, replyWeight);
        };
        int refined = polish(reply, std::max(1, board.size / 64), deadline, score);
        if (strongest) {
            strongest->push_back(reply);
            const int radius = std::max(1, board.size / 27);
            Point alternate = reply;
            int alternateScore = refined - 30;
            for (const auto& candidate : candidates) {
                if (candidate.first > alternateScore &&
                    distanceSquared(candidate.second, reply) >= radius * radius) {
                    alternate = candidate.second;
                    alternateScore = candidate.first;
                }
            }
            if (distanceSquared(reply, alternate)) {
                refined = std::max(refined, polish(alternate, std::max(1, board.size / 64),
                                                   deadline, score));
                strongest->push_back(alternate);
            }
        }
        if (deadline.expired(0.004)) return incomplete;
        return -refined;
    };
    std::vector<std::pair<int, Point>> ranking;
    const auto screen = [&](Point p, int cutoff) {
        const int score = defend(p, cutoff, nullptr);
        if (score != incomplete) ranking.push_back({score, p});
        return score;
    };
    for (const Move& move : seeds) {
        if (deadline.expired(0.045)) break;
        const int score = screen(move.at, bestScore == incomplete ? incomplete : bestScore - 14);
        if (score > bestScore) {
            bestScore = score;
            best = move;
        }
    }
    if (bestScore != incomplete && !deadline.expired(0.045)) {
        int incumbent = bestScore;
        polish(best.at, std::max(1, board.size / 68), deadline, [&](Point q) {
            const int value = screen(q, incumbent - 14);
            incumbent = std::max(incumbent, value);
            return value;
        });
    }
    if (ranking.empty() || deadline.expired(0.65)) return best;

    // Recount the leading defenses on the actual integer board, including a
    // fresh local search for the opponent for each of our nearby moves.
    const PullMap exact(board, fullBoard(board.size), &deadline);
    if (!exact.complete() || deadline.expired(0.090)) return best;
    const auto exactScore = [&](Point p) {
        if (!board.legal(p) || deadline.expired(0.045)) return incomplete;
        std::vector<Point> replies;
        if (defend(p, incomplete, &replies) == incomplete) return incomplete;
        if (replies.empty()) return 0;
        const Move ours{p, ourWeight};
        const Territory rival = exact.territory(false, &ours);
        int value = 0;
        for (Point q : replies) {
            const int refined = polish(q, std::max(1, board.size / 240), deadline, [&](Point at) {
                if (!board.legal(at) || distanceSquared(at, p) < space.spacingSquared) return -1;
                return rival.score(at, replyWeight);
            });
            value = std::max(value, refined);
        }
        return deadline.expired(0.005) ? incomplete : -value;
    };
    std::stable_sort(ranking.begin(), ranking.end(),
                     [](const auto& a, const auto& b) { return a.first > b.first; });
    std::vector<Point> tested{best.at};
    int exactBest = exactScore(best.at);
    for (const auto& entry : ranking) {
        if (tested.size() >= 7 || deadline.expired(0.090)) break;
        const Point p = entry.second;
        if (std::any_of(tested.begin(), tested.end(), [&](Point q) {
                return distanceSquared(p, q) < 36;
            })) continue;
        const int value = exactScore(p);
        tested.push_back(p);
        if (value > exactBest) {
            exactBest = value;
            best.at = p;
        }
    }
    if (exactBest != incomplete && !deadline.expired(0.18)) {
        polish(best.at, std::max(1, board.size / 260), deadline, exactScore);
    }
    return best;
}

struct ReplyLine {
    Choice ourMove;
    std::vector<Choice> replies;
    float estimate;
    float forecast = -2.0f;
};

// Test each likely counter with several of our own follow-ups. Spatial grids and
// weight schedules differ across simulations to avoid relying on one rollout policy.
float investigate(const PairSearch& search, const State& after, const ReplyLine& line,
                  int me, const Deadline& deadline) {
    const Weight plies = after.turns[0] + after.turns[1];
    const auto detail = plies <= 12 ? SearchDepth::Reply : SearchDepth::Rollout;
    const int width = plies <= 12 ? 3 : 2;
    const std::array<int, 5> styles{0, 2, 1, 0, 2};
    auto replies = line.replies;
    if (after.actor != me) {
        const Choice immediate = search.immediateReply(after);
        if (immediate.id >= 0 && std::none_of(replies.begin(), replies.end(), [&](const Choice& q) {
                return q.id == immediate.id && q.weight == immediate.weight;
            })) {
            if (replies.size() == styles.size()) replies.pop_back();
            replies.push_back(immediate);
        }
    }
    float worst = 2.0f;
    float sum = 0;
    int count = 0;
    for (const Choice& reply : replies) {
        if (deadline.expired(0.018)) break;
        const State follow = search.play(after, reply);
        const int lattice = count % 3;
        const auto responses = search.rank(follow, detail, width, 0, lattice);
        float best = -2.0f;
        for (const Choice& response : responses) {
            const float value = search.rollout(search.play(follow, response), me, styles[count], lattice);
            best = std::max(best, value);
        }
        if (responses.empty()) best = search.rollout(follow, me, styles[count], lattice);
        worst = std::min(worst, best);
        sum += best;
        ++count;
    }
    return count ? 0.57f * worst + 0.43f * (sum / count) : line.forecast;
}

Move playPair(const Board& board, SearchSpace& space, const Deadline& deadline, Move backup) {
    PairSearch search(board, space, deadline);
    State root = search.root();
    const int me = board.me;
    const Weight plies = root.turns[0] + root.turns[1];
    const bool finalDuel = root.turns[me] == 1 && root.turns[1 - me] == 1;
    auto choices = search.rank(root, SearchDepth::Root, 76);
    if (choices.empty()) return backup;
    if (!deadline.expired(0.25)) {
        std::vector<Point> centers;
        for (const Choice& choice : choices) {
            const Point p = search.move(choice).at;
            if (std::none_of(centers.begin(), centers.end(), [&](Point other) {
                    return distanceSquared(p, other) < (board.size / 24) * (board.size / 24);
                })) centers.push_back(p);
            if (centers.size() >= 22) break;
        }
        space.densify(board, centers);
        root = search.root();
        const auto refined = search.rank(root, SearchDepth::Root, 76);
        if (!refined.empty()) choices = refined;
    }
    Move best = search.move(choices.front());
    std::vector<ReplyLine> lines;
    for (const Choice& choice : choices) {
        if (deadline.expired(0.025)) break;
        const State after = search.play(root, choice);
        auto replies = search.rank(after, SearchDepth::Reply, plies > 6 ? 5 : 1);
        if (replies.empty()) continue;
        const float response = (after.actor == me ? 1.0f : -1.0f) * replies.front().value;
        const float estimate = finalDuel ? response : 0.22f * choice.value + 0.78f * response;
        lines.push_back({choice, std::move(replies), estimate});
    }
    if (lines.empty()) return best;
    std::stable_sort(lines.begin(), lines.end(),
                     [](const ReplyLine& a, const ReplyLine& b) { return a.estimate > b.estimate; });
    best = search.move(lines.front().ourMove);

    if (finalDuel) {
        std::vector<Move> seeds;
        for (std::size_t i = 0; i < std::min<std::size_t>(24, lines.size()); ++i) {
            seeds.push_back(search.move(lines[i].ourMove));
        }
        return lastDuel(board, space, deadline, seeds);
    }

    float bestScore = -2.0f;
    if (plies <= 6) {
        constexpr float staticPart = 0.07f;
        const std::size_t limit = std::min<std::size_t>(lines.size(), 33);
        for (std::size_t i = 0; i < limit && !deadline.expired(0.045); ++i) {
            const ReplyLine& line = lines[i];
            const State after = search.play(root, line.ourMove);
            const float alpha = (bestScore - staticPart * line.estimate) / (1.0f - staticPart);
            const float leaf = search.minimax(after, me, static_cast<int>(plies - 1), alpha, 2.0f);
            const float score = staticPart * line.estimate + (1.0f - staticPart) * leaf;
            if (deadline.expired(0.012)) break;
            if (score > bestScore) {
                bestScore = score;
                best = search.move(line.ourMove);
            }
        }
        return best;
    }

    const std::size_t shallow = std::min<std::size_t>(lines.size(), plies > 22 ? 38 : 55);
    for (std::size_t i = 0; i < shallow && !deadline.expired(0.045); ++i) {
        ReplyLine& line = lines[i];
        const State after = search.play(root, line.ourMove);
        const float first = search.rollout(search.play(after, line.replies[0]), me, 0, 0);
        const Choice& alternate = line.replies[std::min<std::size_t>(1, line.replies.size() - 1)];
        const float second = search.rollout(search.play(after, alternate), me, 2, 1);
        if (deadline.expired(0.015)) break;
        const float terminal = 0.65f * std::min(first, second) + 0.35f * std::max(first, second);
        line.forecast = 0.28f * line.estimate + 0.72f * terminal;
        if (line.forecast > bestScore) {
            bestScore = line.forecast;
            best = search.move(line.ourMove);
        }
    }
    std::stable_sort(lines.begin(), lines.end(),
                     [](const ReplyLine& a, const ReplyLine& b) { return a.forecast > b.forecast; });
    bestScore = -2.0f;
    const std::size_t deep = std::min<std::size_t>(lines.size(), plies > 22 ? 12 : 21);
    for (std::size_t i = 0; i < deep && !deadline.expired(0.050); ++i) {
        const ReplyLine& line = lines[i];
        if (line.forecast == -2.0f) break;
        const State after = search.play(root, line.ourMove);
        const float terminal = investigate(search, after, line, me, deadline);
        if (deadline.expired(0.015)) break;
        const float score = 0.23f * line.estimate + 0.17f * line.forecast + 0.60f * terminal;
        if (score > bestScore) {
            bestScore = score;
            best = search.move(line.ourMove);
        }
    }
    return best;
}

Move playMelee(const Board& board, const SearchSpace& space, const Deadline& deadline, Move best) {
    const int count = static_cast<int>(space.points.size());
    const float unit = 1.0f / static_cast<float>(board.totalWeight);
    std::vector<std::vector<float>> pull(board.players, space.forecast);
    for (int p = 0; p < board.players; ++p) {
        const float future = board.turns[p] > 0 ? board.reserve[p] * unit : 0;
        for (float& value : pull[p]) value *= future;
    }
    for (const Stone& s : board.stones) {
        for (int i = 0; i < count; ++i) {
            pull[s.player][i] += s.move.weight * unit * space.pull(s.move.at, space.points[i]);
        }
    }
    std::vector<float> margin(count);
    for (int i = 0; i < count; ++i) {
        float rival = 0;
        for (int p = 0; p < board.players; ++p) {
            if (p != board.me) rival = std::max(rival, pull[p][i]);
        }
        margin[i] = pull[board.me][i] - rival;
    }
    const auto weights = weightChoices(board.reserve[board.me], board.turns[board.me], SearchDepth::Root);
    float bestScore = -2.0f;
    for (std::size_t id = 0; id < space.locations.size(); ++id) {
        if (deadline.expired(0.006)) break;
        for (Weight w : weights) {
            const float width = 0.13f + 1.8f * (board.reserve[board.me] - w) * unit;
            const float score = control(margin.data(), space.kernel(static_cast<int>(id)),
                                        count, w * unit, width);
            if (score > bestScore) {
                bestScore = score;
                best = {space.locations[id], w};
            }
        }
    }
    return best;
}

Move choose(const Board& board) {
    if (!board.reserve[board.me]) return {{0, 0}, 0};
    const Deadline deadline(board);
    const Move backup = board.fallback();
    if (backup.weight == 0 || deadline.expired(0.045)) return backup;
    SearchSpace space(board, backup.at);
    if (deadline.expired(0.010)) return backup;
    bool rivalWillMove = false;
    for (int p = 0; p < board.players; ++p) {
        rivalWillMove |= p != board.me && board.turns[p] > 0 && board.reserve[p] > 0;
    }
    if (board.turns[board.me] == 1 && !rivalWillMove) {
        return lastMove(board, space, deadline, backup);
    }
    return board.players == 2 ? playPair(board, space, deadline, backup) :
                               playMelee(board, space, deadline, backup);
}

}  // namespace

int main() {
    std::ios::sync_with_stdio(false);
    std::cin.tie(nullptr);
    std::string line;
    while (std::getline(std::cin, line)) {
        const std::string type(jsonValue(line, "type"));
        if (type.compare(0, 7, "\"start\"") == 0) {
            std::cout << "{\"ready\":true}\n" << std::flush;
        } else if (type.compare(0, 6, "\"move\"") == 0) {
            const Move move = choose(Board(line));
            std::cout << "{\"x\":" << move.at.x << ",\"y\":" << move.at.y
                      << ",\"weight\":" << move.weight << "}\n" << std::flush;
        }
    }
}
