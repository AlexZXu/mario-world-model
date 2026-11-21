#pragma once
#include <random>
// Fixed algorithm and output range, independent of the platform's libc rand().
namespace MarioRandom {
inline std::mt19937 generator{0};
inline void seed(unsigned value) { generator.seed(value); }
inline int next() { return int(generator() & 0x7fffffffU); }
inline constexpr int maximum = 0x7fffffff;
}
