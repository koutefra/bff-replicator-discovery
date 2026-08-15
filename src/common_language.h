/*
 * Copyright 2024 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <stdint.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <random>
#include <string>
#include <vector>

#include "common.h"
#ifdef __CUDACC__
#include <cuda_device_runtime_api.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>

#define CUCHECK(op)                                                           \
  {                                                                           \
    cudaError_t cudaerr = op;                                                 \
    if (cudaerr != cudaSuccess) {                                             \
      printf("%s failed with error: %s\n", #op, cudaGetErrorString(cudaerr)); \
      exit(1);                                                                \
    }                                                                         \
  }

constexpr size_t kWarpSize = 32;

__inline__ __device__ uint64_t warpReduceSum(uint64_t val) {
  for (int offset = warpSize / 2; offset > 0; offset /= 2)
    val += __shfl_down_sync(~0, val, offset);
  return val;
}

__inline__ __device__ size_t GetIndex() {
  return threadIdx.x + blockIdx.x * blockDim.x;
}

__inline__ __device__ void IncreaseInsnCount(unsigned long long count,
                                             unsigned long long *storage) {
  size_t index = GetIndex();
  size_t warp_ops = warpReduceSum(count);
  if (index % kWarpSize == 0) {
    atomicAdd(storage, warp_ops);
  }
}

inline void Synchronize() { CUCHECK(cudaDeviceSynchronize()); }

template <typename T>
struct DeviceMemory {
  T *data;
  DeviceMemory(size_t size) { CUCHECK(cudaMalloc(&data, size * sizeof(T))); }
  void Write(const T *host, size_t count) {
    CUCHECK(cudaMemcpy(data, host, count * sizeof(T), cudaMemcpyHostToDevice));
  }
  void Read(T *host, size_t count) {
    CUCHECK(cudaMemcpy(host, data, count * sizeof(T), cudaMemcpyDeviceToHost));
  }
  T *Get() { return data; }
  ~DeviceMemory() { CUCHECK(cudaFree(data)); }
  DeviceMemory(DeviceMemory &) = delete;
};

#define RUN(grid, block, fun, ...) fun<<<grid, block>>>(__VA_ARGS__)

#else
#define __device__
#define __host__
#define __global__

inline size_t &IndexThreadLocal() {
  thread_local size_t index;
  return index;
}

inline size_t GetIndex() { return IndexThreadLocal(); }

inline void IncreaseInsnCount(unsigned long long count,
                              unsigned long long *storage) {
  __atomic_add_fetch(storage, count, __ATOMIC_RELAXED);
}

inline void Synchronize() {}

template <typename T>
struct DeviceMemory {
  T *data;
  DeviceMemory(size_t size) { data = (T *)malloc(size * sizeof(T)); }
  void Write(const T *host, size_t count) {
    memcpy(data, host, count * sizeof(T));
  }
  void Read(T *host, size_t count) { memcpy(host, data, count * sizeof(T)); }
  T *Get() { return data; }
  ~DeviceMemory() { free(data); }
  DeviceMemory(DeviceMemory &) = delete;
};

#define RUN(grid, block, fun, ...)                                            \
  _Pragma("omp parallel for") for (size_t _threadcnt = 0;                     \
                                   _threadcnt < grid * block; _threadcnt++) { \
    IndexThreadLocal() = _threadcnt;                                          \
    fun(__VA_ARGS__);                                                         \
  }

#endif

#define CHECK(op)                 \
  if (!(op)) {                    \
    printf("%s is false\n", #op); \
    exit(1);                      \
  }

inline __device__ __host__ uint64_t SplitMix64(uint64_t seed) {
  uint64_t z = seed + 0x9e3779b97f4a7c15;
  z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9;
  z = (z ^ (z >> 27)) * 0x94d049bb133111eb;
  return z ^ (z >> 31);
}

template <typename Language>
__global__ void InitPrograms(size_t seed, size_t num_programs,
                             uint8_t *programs, bool zero_init,
                             const uint64_t *init_byte_cdf) {
  size_t index = GetIndex();
  auto prog = programs + index * kSingleTapeSize;
  if (index >= num_programs) return;
  if (zero_init) {
    for (size_t i = 0; i < kSingleTapeSize; i++) {
      prog[i] = 0;
    }
  } else if (init_byte_cdf != nullptr) {
    for (size_t i = 0; i < kSingleTapeSize; i++) {
      uint64_t draw =
          SplitMix64(kSingleTapeSize * num_programs * seed +
                     kSingleTapeSize * index + i) >>
          1;
      size_t lo = 0;
      size_t hi = 256;
      while (lo < hi) {
        size_t mid = lo + (hi - lo) / 2;
        if (draw < init_byte_cdf[mid]) {
          hi = mid;
        } else {
          lo = mid + 1;
        }
      }
      prog[i] = static_cast<uint8_t>(lo < 256 ? lo : 255);
    }
  } else {
    for (size_t i = 0; i < kSingleTapeSize; i++) {
      prog[i] = SplitMix64(kSingleTapeSize * num_programs * seed +
                           kSingleTapeSize * index + i) %
                256;
    }
  }
}

template <typename Language>
__global__ void MutateAndRunPrograms(uint8_t *programs,
                                     const uint32_t *shuf_idx, size_t seed,
                                     uint32_t mutation_prob,
                                     unsigned long long *insn_count,
                                     unsigned long long *step_count_sum,
                                     unsigned long long *termination_counts,
                                     size_t num_programs, size_t num_indices) {
  size_t index = GetIndex();
  uint8_t tape[2 * kSingleTapeSize] = {};
  if (2 * index >= num_programs) return;
  uint32_t p1 = shuf_idx[2 * index];
  uint32_t p2 = shuf_idx[2 * index + 1];
  for (size_t i = 0; i < kSingleTapeSize; i++) {
    tape[i] = programs[p1 * kSingleTapeSize + i];
    tape[i + kSingleTapeSize] = programs[p2 * kSingleTapeSize + i];
  }
  for (size_t i = 0; i < 2 * kSingleTapeSize; i++) {
    uint64_t rng =
        SplitMix64((num_programs * seed + index) * kSingleTapeSize * 2 + i);
    uint8_t repl = rng & 0xFF;
    uint64_t prob_rng = (rng >> 8) & ((1ULL << 30) - 1);
    if (prob_rng < mutation_prob) {
      tape[i] = repl;
    }
  }
  bool debug = false;
  size_t ops = 0;
  size_t step_count = 0;
  uint8_t termination_reason = kTerminationStepCap;
  if (index < num_indices) {
    ops = Language::Evaluate(tape, 8 * 1024, debug, &step_count,
                             &termination_reason);
    if (termination_reason >= kNumTerminationReasons) {
      termination_reason = kTerminationIpOutOfBounds;
    }
  }
  for (size_t i = 0; i < kSingleTapeSize; i++) {
    programs[p1 * kSingleTapeSize + i] = tape[i];
    programs[p2 * kSingleTapeSize + i] = tape[i + kSingleTapeSize];
  }
  if (index < num_indices) {
    IncreaseInsnCount(ops, insn_count);
    IncreaseInsnCount(step_count, step_count_sum);
    IncreaseInsnCount(1, termination_counts + termination_reason);
  }
}

template <typename Language>
__global__ void MutateAndRunProgramsRandomPartner(
    uint8_t *programs, size_t seed, size_t epoch, uint32_t mutation_prob,
    unsigned long long *insn_count, unsigned long long *step_count_sum,
    unsigned long long *termination_counts, size_t num_programs) {
  size_t index = GetIndex();
  if (index >= num_programs) return;

  uint8_t tape[2 * kSingleTapeSize] = {};
  uint8_t random_prog[kSingleTapeSize] = {};

  const uint8_t *program_a = programs + index * kSingleTapeSize;

  // --- Construct a strong base seed from (global seed, epoch) ---
  uint64_t base_seed =
      SplitMix64(SplitMix64((uint64_t)seed) ^ SplitMix64((uint64_t)epoch));

  // --- Stream 1: random partner bytes (depends on index, but independent of mutation) ---
  uint64_t partner_seed =
      SplitMix64(base_seed ^ SplitMix64((uint64_t)index));

  for (size_t i = 0; i < kSingleTapeSize; ++i) {
    uint64_t x = SplitMix64(partner_seed + i);
    random_prog[i] = x & 0xFF;
  }

  // Decide whether A is first or second using a separate bit from the same stream
  bool a_first =
      (SplitMix64(partner_seed ^ 0x9e3779b97f4a7c15ULL) & 1ULL) == 0ULL;
  size_t a_offset = a_first ? 0 : kSingleTapeSize;
  uint8_t *a_half = tape + a_offset;
  uint8_t *r_half = tape + (a_first ? kSingleTapeSize : 0);

  for (size_t i = 0; i < kSingleTapeSize; ++i) {
    a_half[i] = program_a[i];
    r_half[i] = random_prog[i];
  }

  // --- Stream 2: mutation RNG (separate from partner_seed) ---
  uint64_t mutation_seed =
      SplitMix64(base_seed ^ 0x6a09e667f3bcc909ULL);  // different constant

  for (size_t i = 0; i < 2 * kSingleTapeSize; ++i) {
    // Mix index and position into the mutation stream
    uint64_t x = SplitMix64(mutation_seed ^
                            ((uint64_t)index << 32) ^ (uint64_t)i);
    uint8_t repl = x & 0xFF;
    uint64_t prob_rng = (x >> 8) & ((1ULL << 30) - 1);
    if (prob_rng < mutation_prob) {
      tape[i] = repl;
    }
  }

  bool debug = false;
  size_t step_count = 0;
  uint8_t termination_reason = kTerminationStepCap;
  size_t ops = Language::Evaluate(tape, 8 * 1024, debug, &step_count,
                                  &termination_reason);
  if (termination_reason >= kNumTerminationReasons) {
    termination_reason = kTerminationIpOutOfBounds;
  }

  for (size_t i = 0; i < kSingleTapeSize; ++i) {
    programs[index * kSingleTapeSize + i] = tape[a_offset + i];
  }

  IncreaseInsnCount(ops, insn_count);
  IncreaseInsnCount(step_count, step_count_sum);
  IncreaseInsnCount(1, termination_counts + termination_reason);
}

template <typename Language>
__global__ void MutateAndRunProgramsStructuredPartner(
    uint8_t *programs, const uint8_t *partner_pool, size_t partner_pool_programs,
    size_t seed, size_t epoch, uint32_t mutation_prob,
    unsigned long long *insn_count, unsigned long long *step_count_sum,
    unsigned long long *termination_counts, size_t num_programs) {
  size_t index = GetIndex();
  if (index >= num_programs) return;

  uint8_t tape[2 * kSingleTapeSize] = {};
  const uint8_t *program_a = programs + index * kSingleTapeSize;

  uint64_t base_seed =
      SplitMix64(SplitMix64((uint64_t)seed) ^ SplitMix64((uint64_t)epoch));
  uint64_t partner_seed = SplitMix64(base_seed ^ SplitMix64((uint64_t)index));
  size_t partner_idx =
      SplitMix64(partner_seed ^ 0x243f6a8885a308d3ULL) % partner_pool_programs;
  const uint8_t *partner_prog = partner_pool + partner_idx * kSingleTapeSize;

  bool a_first =
      (SplitMix64(partner_seed ^ 0x9e3779b97f4a7c15ULL) & 1ULL) == 0ULL;
  size_t a_offset = a_first ? 0 : kSingleTapeSize;
  uint8_t *a_half = tape + a_offset;
  uint8_t *p_half = tape + (a_first ? kSingleTapeSize : 0);

  for (size_t i = 0; i < kSingleTapeSize; ++i) {
    a_half[i] = program_a[i];
    p_half[i] = partner_prog[i];
  }

  uint64_t mutation_seed = SplitMix64(base_seed ^ 0x6a09e667f3bcc909ULL);
  for (size_t i = 0; i < 2 * kSingleTapeSize; ++i) {
    uint64_t x = SplitMix64(mutation_seed ^
                            ((uint64_t)index << 32) ^ (uint64_t)i);
    uint8_t repl = x & 0xFF;
    uint64_t prob_rng = (x >> 8) & ((1ULL << 30) - 1);
    if (prob_rng < mutation_prob) {
      tape[i] = repl;
    }
  }

  bool debug = false;
  size_t step_count = 0;
  uint8_t termination_reason = kTerminationStepCap;
  size_t ops = Language::Evaluate(tape, 8 * 1024, debug, &step_count,
                                  &termination_reason);
  if (termination_reason >= kNumTerminationReasons) {
    termination_reason = kTerminationIpOutOfBounds;
  }

  for (size_t i = 0; i < kSingleTapeSize; ++i) {
    programs[index * kSingleTapeSize + i] = tape[a_offset + i];
  }

  IncreaseInsnCount(ops, insn_count);
  IncreaseInsnCount(step_count, step_count_sum);
  IncreaseInsnCount(1, termination_counts + termination_reason);
}


template <typename Language>
__global__ void RunOneProgram(uint8_t *program, size_t stepcount, bool debug) {
  size_t ops = Language::Evaluate(program, stepcount, debug);
  printf("%s", ResetColors());
  printf("ops: %d\n", (int)ops);
  printf("\n");
}

template <typename Language>
__global__ void CheckSelfRep(uint8_t *programs, size_t seed,
                             size_t num_programs, size_t *result, bool debug) {
  size_t index = GetIndex();
  constexpr size_t kNumIters = 13;
  constexpr size_t kNumExtraGens = 4;
  uint8_t tapes[kNumIters][2 * kSingleTapeSize] = {};
  if (index > num_programs) return;
  uint64_t local_seed = SplitMix64(num_programs * seed + index);
  for (size_t i = 0; i < kNumIters; i++) {
    bool eval_debug = false;
    uint8_t noise[kSingleTapeSize];
    for (int j = 0; j < kSingleTapeSize; j++) {
      noise[j] =
          SplitMix64(local_seed ^ SplitMix64((i + 1) * kSingleTapeSize + j)) %
          256;
    }
    uint8_t *tape = &tapes[i][0];
    for (int j = 0; j < kSingleTapeSize; j++) {
      tape[j] = programs[index * kSingleTapeSize + j];
      tape[j + kSingleTapeSize] = noise[j];
    }
    if (debug) {
      size_t separators[1] = {kSingleTapeSize};
      printf("Iteration %lu: before first step\n", i);
      Language::PrintProgram(2 * kSingleTapeSize, tape, 2 * kSingleTapeSize,
                             separators, 1);
    }
    Language::Evaluate(tape, 8 * 1024, eval_debug);
    if (debug) {
      size_t separators[1] = {kSingleTapeSize};
      printf("Iteration %lu: after first step\n", i);
      Language::PrintProgram(2 * kSingleTapeSize, tape, 2 * kSingleTapeSize,
                             separators, 1);
    }

    for (size_t g = 0; g < kNumExtraGens; g++) {
      for (int j = 0; j < kSingleTapeSize; j++) {
        tape[j] = tape[j + kSingleTapeSize];
        tape[j + kSingleTapeSize] = noise[j];
      }
      if (debug) {
        size_t separators[1] = {kSingleTapeSize};
        printf("Iteration %lu: before step %lu\n", i, g + 2);
        Language::PrintProgram(2 * kSingleTapeSize, tape, 2 * kSingleTapeSize,
                               separators, 1);
      }
      Language::Evaluate(tape, 8 * 1024, eval_debug);
      if (debug) {
        size_t separators[1] = {kSingleTapeSize};
        printf("Iteration %lu: after step %lu\n", i, g + 2);
        Language::PrintProgram(2 * kSingleTapeSize, tape, 2 * kSingleTapeSize,
                               separators, 1);
      }
    }
  }
  size_t res[2] = {};
  for (int i = 0; i < 2 * kSingleTapeSize; ++i) {
    for (size_t a = 0; a < kNumIters; a++) {
      size_t count = 1;
      if (i < kSingleTapeSize &&
          tapes[a][i] != programs[index * kSingleTapeSize + i]) {
        continue;
      }
      for (size_t b = a + 1; b < kNumIters; b++) {
        if (tapes[a][i] == tapes[b][i]) count++;
      }
      if (count > kNumIters / 4) {
        res[i / kSingleTapeSize]++;
        break;
      }
    }
  }
  result[index] = res[0] < res[1] ? res[0] : res[1];
}

template <typename Language>
void Simulation<Language>::RunSingleParsedProgram(
    const std::vector<uint8_t> &parsed, size_t stepcount, bool debug) const {
  DeviceMemory<uint8_t> mem(kSingleTapeSize * 2);
  uint8_t zero[2 * kSingleTapeSize] = {};
  memcpy(zero, parsed.data(), parsed.size());
  mem.Write(zero, 2 * kSingleTapeSize);
  Language::PrintProgram(2 * kSingleTapeSize, zero, 2 * kSingleTapeSize,
                         nullptr, 0);

  RUN(1, 1, RunOneProgram<Language>, mem.Get(), stepcount, debug);

  uint8_t final_state[2 * kSingleTapeSize];
  Synchronize();
  mem.Read(final_state, 2 * kSingleTapeSize);
  Language::PrintProgram(2 * kSingleTapeSize, final_state, 2 * kSingleTapeSize,
                         nullptr, 0);
}

template <typename Language>
void Simulation<Language>::RunSingleProgram(std::string program,
                                            size_t stepcount,
                                            bool debug) const {
  RunSingleParsedProgram(Language::Parse(program), stepcount, debug);
}

template <typename Language>
void Simulation<Language>::PrintProgram(size_t pc_pos, const uint8_t *mem,
                                        size_t len, const size_t *separators,
                                        size_t num_separators) const {
  Language::PrintProgram(pc_pos, mem, len, separators, num_separators);
}

template <typename Language>
std::vector<uint8_t> Simulation<Language>::Parse(const std::string& program) {
  return Language::Parse(program);
}


template <typename Language>
size_t Simulation<Language>::EvalSelfrep(std::string program, size_t epoch,
                                         size_t seed, bool debug) {
  std::vector<uint8_t> parsed = Language::Parse(program);
  return EvalParsedSelfrep(parsed, epoch, seed, debug);
}

template <typename Language>
size_t Simulation<Language>::EvalParsedSelfrep(std::vector<uint8_t> &parsed,
                                               size_t epoch, size_t seed,
                                               bool debug) {
  DeviceMemory<uint8_t> mem(kSingleTapeSize);
  uint8_t zero[kSingleTapeSize] = {};
  memcpy(zero, parsed.data(), parsed.size());
  mem.Write(zero, kSingleTapeSize);
  DeviceMemory<size_t> result(1);
  size_t epoch_seed = SplitMix64(SplitMix64(seed) ^ SplitMix64(epoch));
  RUN(1, 1, CheckSelfRep<Language>, mem.Get(), epoch_seed, 1, result.Get(),
      debug);

  Synchronize();
  std::vector<size_t> res(1);
  result.Read(res.data(), 1);
  return res[0];
}

template <typename Language>
void Simulation<Language>::RunSimulation(
    const SimulationParams &params, std::optional<std::string> initial_program,
    std::function<bool(const SimulationState &)> callback) const {
  constexpr size_t kNumThreads = 32;
  size_t num_programs = params.num_programs;

  size_t reset_index = 1;
  size_t epoch = 0;

  FILE *load_file = nullptr;
  if (params.load_from.has_value()) {
    load_file = CheckFopen(params.load_from->c_str(), "r");
    CHECK(fread(&reset_index, sizeof(reset_index), 1, load_file) == 1);
    CHECK(fread(&num_programs, sizeof(num_programs), 1, load_file) == 1);
    CHECK(fread(&epoch, sizeof(epoch), 1, load_file) == 1);
  }

  DeviceMemory<uint8_t> programs(kSingleTapeSize * num_programs);
  DeviceMemory<unsigned long long> insn_count(1);
  DeviceMemory<unsigned long long> step_count_sum(1);
  DeviceMemory<unsigned long long> termination_counts(kNumTerminationReasons);

  if (!params.reinit_each_epoch && !params.random_partner_interaction &&
      !params.structured_partner_interaction) {
    CHECK(num_programs % 2 == 0);
  }

  size_t structured_pool_programs = 0;
  std::unique_ptr<DeviceMemory<uint8_t>> structured_partner_pool;
  if (params.structured_partner_interaction) {
    CHECK(!params.structured_partner_pool.empty());
    CHECK(params.structured_partner_pool.size() % kSingleTapeSize == 0);
    structured_pool_programs = params.structured_partner_pool.size() / kSingleTapeSize;
    structured_partner_pool =
        std::make_unique<DeviceMemory<uint8_t>>(params.structured_partner_pool.size());
    structured_partner_pool->Write(params.structured_partner_pool.data(),
                                   params.structured_partner_pool.size());
  }

  auto seed = [&](size_t seed2) {
    return SplitMix64(SplitMix64(params.seed) ^ SplitMix64(seed2));
  };

  std::unique_ptr<DeviceMemory<uint64_t>> init_byte_cdf;
  if (!params.init_byte_cdf.empty()) {
    CHECK(params.init_byte_cdf.size() == 256);
    init_byte_cdf = std::make_unique<DeviceMemory<uint64_t>>(256);
    init_byte_cdf->Write(params.init_byte_cdf.data(), params.init_byte_cdf.size());
  }

  RUN((num_programs + kNumThreads - 1) / kNumThreads, kNumThreads,
      InitPrograms<Language>, seed(0), num_programs, programs.Get(),
      params.zero_init, init_byte_cdf ? init_byte_cdf->Get() : nullptr);

  if (initial_program.has_value()) {
    std::vector<uint8_t> parsed = Language::Parse(*initial_program);
    programs.Write((const unsigned char *)parsed.data(), parsed.size());
  }

  unsigned long long zero = 0;
  insn_count.Write(&zero, 1);
  step_count_sum.Write(&zero, 1);
  std::array<unsigned long long, kNumTerminationReasons>
      zero_termination_counts = {};
  termination_counts.Write(zero_termination_counts.data(),
                           zero_termination_counts.size());

  unsigned long long total_ops = 0;

  SimulationState state;
  state.soup.reserve(num_programs * kSingleTapeSize + 16);
  state.soup.resize(num_programs * kSingleTapeSize);
  state.replication_per_prog.resize(num_programs);
  state.shuffle_idx.resize(num_programs);
  Language::InitByteColors(state.byte_colors);
  size_t focal_program_count = 0;
  std::vector<uint8_t> focal_previous_soup;
  if (params.focal_analysis) {
    focal_program_count = std::min(params.focal_program_count, num_programs);
    focal_previous_soup.resize(focal_program_count * kSingleTapeSize);
  }

  if (params.save_to.has_value()) {
    CHECK(mkdir(params.save_to->c_str(),
                S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH) != -1 ||
          errno == EEXIST);
  }

  if (load_file) {
    CHECK(fread(state.soup.data(), 1, num_programs * kSingleTapeSize,
                load_file) == num_programs * kSingleTapeSize);
    fclose(load_file);
    programs.Write(state.soup.data(), num_programs * kSingleTapeSize);
  }
  if (params.focal_analysis && !focal_previous_soup.empty()) {
    Synchronize();
    programs.Read(focal_previous_soup.data(), focal_previous_soup.size());
  }

  DeviceMemory<uint32_t> shuf_idx(num_programs);

  std::vector<uint32_t> &s = state.shuffle_idx;

  for (size_t i = 0; i < num_programs; i++) {
    s[i] = i;
  }

  std::vector<uint32_t> shuffle_tmp_buf(num_programs);
  std::vector<char> used_program(num_programs);

  Synchronize();

  auto do_shuffle = [&](uint32_t *begin, uint32_t *end, uint64_t base_seed) {
    size_t len = end - begin;
    for (size_t i = len; i-- > 0;) {
      size_t j = SplitMix64(seed(epoch * len + i)) % (i + 1);
      std::swap(begin[i], begin[j]);
    }
  };

  std::vector<uint8_t> brotlified_data(
      BrotliEncoderMaxCompressedSize(num_programs * kSingleTapeSize));

  size_t num_runs = 0;
  auto start = std::chrono::high_resolution_clock::now();
  auto simulation_start = std::chrono::high_resolution_clock::now();
  for (;; epoch++) {
    if (!params.reinit_each_epoch) {
      if (params.random_partner_interaction) {
        RUN((num_programs + kNumThreads - 1) / kNumThreads, kNumThreads,
            MutateAndRunProgramsRandomPartner<Language>, programs.Get(),
            params.seed, epoch, params.mutation_prob, insn_count.Get(),
            step_count_sum.Get(), termination_counts.Get(), num_programs);
        num_runs += num_programs;
      } else if (params.structured_partner_interaction) {
        RUN((num_programs + kNumThreads - 1) / kNumThreads, kNumThreads,
            MutateAndRunProgramsStructuredPartner<Language>, programs.Get(),
            structured_partner_pool->Get(), structured_pool_programs,
            params.seed, epoch, params.mutation_prob, insn_count.Get(),
            step_count_sum.Get(), termination_counts.Get(), num_programs);
        num_runs += num_programs;
      } else {
        size_t num_indices = num_programs / 2;
        // Shuffle indices.
        if (!params.allowed_interactions.empty()) {
          for (size_t i = 0; i < num_programs; i++) {
            shuffle_tmp_buf[i] = i;
            used_program[i] = false;
          }
          do_shuffle(shuffle_tmp_buf.data(),
                     shuffle_tmp_buf.data() + shuffle_tmp_buf.size(), epoch);
          num_indices = 0;
          for (size_t i : shuffle_tmp_buf) {
            auto &interact = params.allowed_interactions;
            if (interact.size() <= i || interact[i].empty()) {
              continue;
            }
            size_t idx = seed(seed(epoch) ^ seed(i)) % interact[i].size();
            size_t neigh = interact[i][idx];
            if (used_program[i] || used_program[neigh]) {
              continue;
            }
            used_program[i] = used_program[neigh] = true;
            s[num_indices * 2] = i;
            s[num_indices * 2 + 1] = neigh;
            num_indices++;
          }
          size_t idx = num_indices * 2;
          for (size_t i = 0; i < num_programs; i++) {
            if (!used_program[i]) {
              s[idx++] = i;
            }
          }
        } else if (params.permute_programs) {
          for (size_t i = 0; i < num_programs; i++) {
            s[i] = i;
          }
          if (params.fixed_shuffle) {
            size_t flip = epoch & 1;
            size_t max_pow2 = 31 - __builtin_clz(num_programs);
            size_t offset = (1 << (epoch % max_pow2 + 1)) - 1;
            for (size_t i = 0; i < num_programs; i++) {
              s[i] = ((i * offset) % num_programs) ^ flip;
            }
          } else {
            do_shuffle(s.data(), s.data() + s.size(), epoch);
          }
        } else if (epoch % 2 == 1) {
          for (size_t i = 0; i < num_programs; i++) {
            s[i] = i;
          }
        } else {
          for (size_t i = 0; i < num_programs; i++) {
            s[i] = i == 0 ? num_programs - 1 : i - 1;
          }
        }

        shuf_idx.Write(s.data(), num_programs);

        RUN((num_programs + 2 * kNumThreads - 1) / (2 * kNumThreads),
            kNumThreads, MutateAndRunPrograms<Language>, programs.Get(),
            shuf_idx.Get(), seed(epoch), params.mutation_prob,
            insn_count.Get(), step_count_sum.Get(), termination_counts.Get(),
            num_programs, num_indices);
        num_runs += num_indices;
      }
    } else {
      RUN((num_programs + kNumThreads - 1) / kNumThreads, kNumThreads,
          InitPrograms<Language>, seed(reset_index), num_programs,
          programs.Get(), params.zero_init,
          init_byte_cdf ? init_byte_cdf->Get() : nullptr);
      reset_index++;
    }

    if (epoch % params.callback_interval == 0) {
      auto stop = std::chrono::high_resolution_clock::now();
      Synchronize();
      unsigned long long insn;
      unsigned long long steps;
      std::array<unsigned long long, kNumTerminationReasons> term_counts = {};
      insn_count.Read(&insn, 1);
      step_count_sum.Read(&steps, 1);
      termination_counts.Read(term_counts.data(), term_counts.size());
      total_ops += insn;
      programs.Read(state.soup.data(), num_programs * kSingleTapeSize);
      Synchronize();
      if (params.focal_analysis) {
        state.has_hamming_stats = true;
        if (focal_program_count == 0) {
          state.mean_hamming = 0.0f;
          state.median_hamming = 0.0f;
          state.p90_hamming = 0.0f;
          state.max_hamming = 0;
        } else {
          std::array<size_t, kSingleTapeSize + 1> hamming_hist = {};
          size_t hamming_sum = 0;
          for (size_t i = 0; i < focal_program_count; ++i) {
            uint8_t *prev = focal_previous_soup.data() + i * kSingleTapeSize;
            const uint8_t *current = state.soup.data() + i * kSingleTapeSize;
            size_t hamming = 0;
            for (size_t j = 0; j < kSingleTapeSize; ++j) {
              if (prev[j] != current[j]) {
                hamming++;
              }
              prev[j] = current[j];
            }
            hamming_hist[hamming]++;
            hamming_sum += hamming;
          }
          auto order_stat_from_hist = [&](size_t rank_one_based) -> size_t {
            size_t seen = 0;
            for (size_t h = 0; h < hamming_hist.size(); ++h) {
              seen += hamming_hist[h];
              if (seen >= rank_one_based) {
                return h;
              }
            }
            return kSingleTapeSize;
          };
          if (focal_program_count % 2 == 1) {
            state.median_hamming =
                order_stat_from_hist(focal_program_count / 2 + 1);
          } else {
            size_t left = order_stat_from_hist(focal_program_count / 2);
            size_t right = order_stat_from_hist(focal_program_count / 2 + 1);
            state.median_hamming = (left + right) * 0.5f;
          }
          size_t p90_rank = (9 * focal_program_count + 9) / 10;
          state.p90_hamming = order_stat_from_hist(p90_rank);
          size_t max_hamming = 0;
          for (size_t h = hamming_hist.size(); h-- > 0;) {
            if (hamming_hist[h] > 0) {
              max_hamming = h;
              break;
            }
          }
          state.mean_hamming = hamming_sum * 1.0f / focal_program_count;
          state.max_hamming = max_hamming;
        }
      } else {
        state.has_hamming_stats = false;
        state.mean_hamming = 0.0f;
        state.median_hamming = 0.0f;
        state.p90_hamming = 0.0f;
        state.max_hamming = 0;
      }
      size_t brotli_size = brotlified_data.size();
      BrotliEncoderCompress(2, 24, BROTLI_MODE_GENERIC, state.soup.size(),
                            state.soup.data(), &brotli_size,
                            brotlified_data.data());
      float elapsed_s =
          std::chrono::duration_cast<std::chrono::microseconds>(stop - start)
              .count() *
          1e-6;
      float mops_s = insn * 1.0 / elapsed_s * 1e-6;
      float sim_elapsed_s =
          std::chrono::duration_cast<std::chrono::microseconds>(
              stop - simulation_start)
              .count() *
          1e-6;

      size_t counts[256] = {};
      for (auto c : state.soup) {
        counts[c]++;
      }

      std::vector<uint8_t> sorted(256);
      double h0 = 0;
      for (size_t i = 0; i < 256; i++) {
        sorted[i] = i;
        double frac = counts[i] * 1.0 / state.soup.size();
        h0 -= counts[i] ? frac * std::log2(frac) : 0.0;
      }
      std::sort(sorted.begin(), sorted.end(), [&](uint8_t a, uint8_t b) {
        return std::make_pair(counts[b], b) < std::make_pair(counts[a], a);
      });

      auto count_mapped_symbol = [&](char symbol) {
        size_t total = 0;
        for (size_t i = 0; i < 256; ++i) {
          if (counts[i] == 0) continue;
          char chmem[32];
          const char *mapped = Language::MapChar((uint8_t)i, chmem);
          if (mapped[0] == symbol && mapped[1] == 0) {
            total += counts[i];
          }
        }
        return total;
      };

      double brotli_bpb = brotli_size * 8.0 / (num_programs * kSingleTapeSize);

      state.elapsed_s = sim_elapsed_s;
      state.total_ops = total_ops;
      state.mops_s = mops_s;
      state.epoch = epoch + 1;
      state.ops_per_run =
          num_runs ? insn * 1.0 / num_runs : 0.0;
      state.brotli_size = brotli_size;
      state.brotli_bpb = brotli_bpb;
      state.bytes_per_prog = brotli_size * 1.0 / num_programs;
      state.h0 = h0;
      state.higher_entropy = h0 - brotli_bpb;
      state.mean_step_count_per_interaction =
          num_runs ? steps * 1.0 / num_runs : 0.0;
      state.frac_term_step_cap =
          num_runs ? term_counts[kTerminationStepCap] * 1.0 / num_runs : 0.0;
      state.frac_term_ip_out_of_bounds =
          num_runs ? term_counts[kTerminationIpOutOfBounds] * 1.0 / num_runs
                   : 0.0;
      state.frac_term_bracket_mismatch =
          num_runs ? term_counts[kTerminationBracketMismatch] * 1.0 / num_runs
                   : 0.0;
      state.count_op_lt = count_mapped_symbol('<');
      state.count_op_gt = count_mapped_symbol('>');
      state.count_op_lbrace = count_mapped_symbol('{');
      state.count_op_rbrace = count_mapped_symbol('}');
      state.count_op_plus = count_mapped_symbol('+');
      state.count_op_minus = count_mapped_symbol('-');
      state.count_op_dot = count_mapped_symbol('.');
      state.count_op_comma = count_mapped_symbol(',');
      state.count_op_lbracket = count_mapped_symbol('[');
      state.count_op_rbracket = count_mapped_symbol(']');
      state.count_zero = count_mapped_symbol('0');

      for (size_t i = 0; i < state.frequent_bytes.size(); i++) {
        uint8_t c = sorted[i];
        char chmem[32];
        state.frequent_bytes[i].first = Language::MapChar(c, chmem);
        state.frequent_bytes[i].second =
            counts[(int)c] * 1.0 / state.soup.size();
      }
      for (size_t i = 0; i < state.uncommon_bytes.size(); i++) {
        uint8_t c = sorted[256 - state.uncommon_bytes.size() + i];
        char chmem[32];
        state.uncommon_bytes[i].first = Language::MapChar(c, chmem);
        state.uncommon_bytes[i].second =
            counts[(int)c] * 1.0 / state.soup.size();
      }

      if (params.eval_selfrep) {
        DeviceMemory<size_t> result(num_programs);
        RUN(num_programs / kNumThreads, kNumThreads, CheckSelfRep<Language>,
            programs.Get(), seed(epoch), num_programs, result.Get(), false);
        Synchronize();
        result.Read(state.replication_per_prog.data(), num_programs);
      }
      if (params.save_to.has_value() && (epoch % params.save_interval == 0)) {
        std::vector<char> save_path(params.save_to->size() + 20);
        snprintf(save_path.data(), save_path.size(), "%s/%010zu.dat",
                 params.save_to->c_str(), epoch);
        FILE *f = CheckFopen(save_path.data(), "w");
        size_t epoch_to_save = epoch + 1;
        fwrite(&reset_index, sizeof(reset_index), 1, f);
        fwrite(&num_programs, sizeof(num_programs), 1, f);
        fwrite(&epoch_to_save, sizeof(epoch), 1, f);
        fwrite(state.soup.data(), 1, state.soup.size(), f);
        fclose(f);
      }
      if (callback(state)) {
        break;
      }
      num_runs = 0;
      start = std::chrono::high_resolution_clock::now();
      insn_count.Write(&zero, 1);
      step_count_sum.Write(&zero, 1);
      termination_counts.Write(zero_termination_counts.data(),
                               zero_termination_counts.size());
    }

    if (!params.reinit_each_epoch && params.reset_interval.has_value() &&
        epoch % *params.reset_interval == 0) {
      RUN(num_programs / kNumThreads, kNumThreads, InitPrograms<Language>,
          seed(reset_index), num_programs, programs.Get(), params.zero_init,
          init_byte_cdf ? init_byte_cdf->Get() : nullptr);
      reset_index++;
    }
  }
}
