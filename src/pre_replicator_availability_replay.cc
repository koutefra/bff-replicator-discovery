#include <stdint.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <optional>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "bff.inc.h"

namespace fs = std::filesystem;

namespace {

const char *Bff::name() { return "bff_noheads"; }

constexpr size_t kThreads = 32;
constexpr size_t kReplayStepCap = 8 * 1024;
constexpr size_t kDefaultNumPrograms = 1u << 17;
constexpr size_t kDefaultEpochs = 16000;
constexpr size_t kDefaultScoreThreshold = 60;
constexpr size_t kDefaultEpochStride = 100;

struct FixedTape {
  size_t fixed_id = 0;
  std::string fixed_label;
  size_t bracket_position = 0;
  uint8_t inert_byte = 0;
  std::string fixed_hex;
  std::string fixed_strip;
  std::array<uint8_t, kSingleTapeSize> bytes = {};
};

struct Config {
  std::string regime;
  size_t seed = 0;
  size_t num_programs = kDefaultNumPrograms;
  size_t epochs = kDefaultEpochs;
  size_t score_threshold = kDefaultScoreThreshold;
  size_t epoch_stride = kDefaultEpochStride;
  size_t validation_program_limit = 0;
  size_t validation_fixed_limit = 0;
  std::string fixed_manifest;
  std::string output_dir;
  std::optional<std::string> gpu_label;
};

struct FixedStats {
  size_t fixed_id = 0;
  std::string fixed_label;
  size_t bracket_position = 0;
  uint64_t total_hits = 0;
  uint64_t total_tested = 0;
};

struct Outputs {
  FILE *epoch_file = nullptr;
  FILE *hit_file = nullptr;
  FILE *log_file = nullptr;
  fs::path fixed_summary_path;
};

std::string CsvEscape(const std::string &value) {
  std::string out;
  out.reserve(value.size());
  for (char c : value) {
    if (c == '"') {
      out.push_back('"');
      out.push_back('"');
    } else {
      out.push_back(c);
    }
  }
  return out;
}

std::vector<std::string> ParseCsvLine(const std::string &line) {
  std::vector<std::string> fields;
  std::string cur;
  bool in_quotes = false;
  for (size_t i = 0; i < line.size(); ++i) {
    char c = line[i];
    if (in_quotes) {
      if (c == '"') {
        if (i + 1 < line.size() && line[i + 1] == '"') {
          cur.push_back('"');
          ++i;
        } else {
          in_quotes = false;
        }
      } else {
        cur.push_back(c);
      }
    } else {
      if (c == '"') {
        in_quotes = true;
      } else if (c == ',') {
        fields.push_back(cur);
        cur.clear();
      } else {
        cur.push_back(c);
      }
    }
  }
  fields.push_back(cur);
  return fields;
}

std::array<uint8_t, kSingleTapeSize> HexToTape(const std::string &hex) {
  if (hex.size() != 2 * kSingleTapeSize) {
    fprintf(stderr, "Expected 128 hex chars, got %zu\n", hex.size());
    exit(1);
  }
  std::array<uint8_t, kSingleTapeSize> out = {};
  for (size_t i = 0; i < kSingleTapeSize; ++i) {
    char tmp[3] = {hex[2 * i], hex[2 * i + 1], 0};
    out[i] = static_cast<uint8_t>(strtoul(tmp, nullptr, 16));
  }
  return out;
}

std::optional<std::string> GetArgValue(int argc, char **argv,
                                       const std::string &key) {
  for (int i = 1; i < argc; ++i) {
    if (std::string(argv[i]) == key && i + 1 < argc) {
      return std::string(argv[i + 1]);
    }
  }
  return std::nullopt;
}

void RequireDirEmptyOrCreate(const fs::path &path) {
  if (fs::exists(path)) {
    if (!fs::is_directory(path)) {
      fprintf(stderr, "Output path exists and is not a directory: %s\n",
              path.string().c_str());
      exit(1);
    }
    if (!fs::is_empty(path)) {
      fprintf(stderr,
              "Output directory already exists and is not empty: %s\n",
              path.string().c_str());
      exit(1);
    }
  } else {
    fs::create_directories(path);
  }
}

Config ParseArgs(int argc, char **argv) {
  Config cfg;
  auto regime = GetArgValue(argc, argv, "--regime");
  auto seed = GetArgValue(argc, argv, "--seed");
  auto output_dir = GetArgValue(argc, argv, "--output_dir");
  auto fixed_manifest = GetArgValue(argc, argv, "--fixed_manifest");
  if (!regime.has_value() || !seed.has_value() || !output_dir.has_value() ||
      !fixed_manifest.has_value()) {
    fprintf(stderr,
            "Required args: --regime {N|R} --seed <int> --output_dir <dir> "
            "--fixed_manifest <csv>\n");
    exit(1);
  }
  cfg.regime = *regime;
  if (cfg.regime != "N" && cfg.regime != "R") {
    fprintf(stderr, "--regime must be N or R\n");
    exit(1);
  }
  cfg.seed = strtoull(seed->c_str(), nullptr, 10);
  cfg.output_dir = *output_dir;
  cfg.fixed_manifest = *fixed_manifest;
  if (auto value = GetArgValue(argc, argv, "--num_programs")) {
    cfg.num_programs = strtoull(value->c_str(), nullptr, 10);
  }
  if (auto value = GetArgValue(argc, argv, "--epochs")) {
    cfg.epochs = strtoull(value->c_str(), nullptr, 10);
  }
  if (auto value = GetArgValue(argc, argv, "--score_threshold")) {
    cfg.score_threshold = strtoull(value->c_str(), nullptr, 10);
  }
  if (auto value = GetArgValue(argc, argv, "--epoch_stride")) {
    cfg.epoch_stride = strtoull(value->c_str(), nullptr, 10);
  }
  if (auto value = GetArgValue(argc, argv, "--validation_program_limit")) {
    cfg.validation_program_limit = strtoull(value->c_str(), nullptr, 10);
  }
  if (auto value = GetArgValue(argc, argv, "--validation_fixed_limit")) {
    cfg.validation_fixed_limit = strtoull(value->c_str(), nullptr, 10);
  }
  if (auto value = GetArgValue(argc, argv, "--gpu_label")) {
    cfg.gpu_label = *value;
  }
  if (cfg.num_programs == 0 || cfg.epochs == 0 || cfg.epoch_stride == 0) {
    fprintf(stderr,
            "--num_programs, --epochs, and --epoch_stride must be positive\n");
    exit(1);
  }
  return cfg;
}

std::vector<FixedTape> LoadFixedManifest(const fs::path &path,
                                         size_t fixed_limit) {
  FILE *f = CheckFopen(path.string().c_str(), "r");
  char *line = nullptr;
  size_t cap = 0;
  ssize_t line_len = getline(&line, &cap, f);
  if (line_len <= 0) {
    fprintf(stderr, "Empty fixed manifest: %s\n", path.string().c_str());
    exit(1);
  }
  std::vector<std::string> header = ParseCsvLine(std::string(line, line_len));
  std::unordered_map<std::string, size_t> index;
  for (size_t i = 0; i < header.size(); ++i) {
    index[header[i]] = i;
  }
  auto field = [&](const std::vector<std::string> &row, const char *name)
      -> const std::string & {
    auto it = index.find(name);
    if (it == index.end() || it->second >= row.size()) {
      fprintf(stderr, "Missing field %s in %s\n", name, path.string().c_str());
      exit(1);
    }
    return row[it->second];
  };

  std::vector<FixedTape> fixed_tapes;
  while ((line_len = getline(&line, &cap, f)) > 0) {
    std::vector<std::string> row = ParseCsvLine(std::string(line, line_len));
    if (row.empty() || row[0].empty()) {
      continue;
    }
    FixedTape fixed;
    fixed.fixed_id = strtoull(field(row, "fixed_id").c_str(), nullptr, 10);
    fixed.fixed_label = field(row, "fixed_label");
    fixed.bracket_position =
        strtoull(field(row, "bracket_position").c_str(), nullptr, 10);
    fixed.inert_byte =
        static_cast<uint8_t>(strtoull(field(row, "inert_byte").c_str(), nullptr,
                                      10));
    fixed.fixed_hex = field(row, "fixed_hex");
    fixed.fixed_strip = field(row, "fixed_strip");
    fixed.bytes = HexToTape(fixed.fixed_hex);
    fixed_tapes.push_back(fixed);
    if (fixed_limit > 0 && fixed_tapes.size() >= fixed_limit) {
      break;
    }
  }
  free(line);
  fclose(f);
  if (fixed_tapes.empty()) {
    fprintf(stderr, "No fixed tapes loaded from %s\n", path.string().c_str());
    exit(1);
  }
  return fixed_tapes;
}

uint64_t SimulationSeed(size_t run_seed, size_t seed2) {
  return SplitMix64(SplitMix64(run_seed) ^ SplitMix64(seed2));
}

uint64_t PreReplicatorScoreSeed(size_t run_seed, size_t logical_epoch) {
  return SplitMix64(SplitMix64(run_seed) ^ SplitMix64(logical_epoch - 1));
}

void BuildReplayShuffle(size_t run_seed, size_t internal_epoch,
                        std::vector<uint32_t> *indices) {
  for (size_t i = 0; i < indices->size(); ++i) {
    (*indices)[i] = i;
  }
  size_t len = indices->size();
  for (size_t i = len; i-- > 0;) {
    size_t j =
        SplitMix64(SimulationSeed(run_seed, internal_epoch * len + i)) % (i + 1);
    std::swap((*indices)[i], (*indices)[j]);
  }
}

template <typename Language>
__global__ void ReplayFixedRightBatchKernel(const uint8_t *left_programs,
                                            const uint8_t *right_fixed,
                                            uint8_t *right_offspring,
                                            size_t active_programs,
                                            size_t num_fixed) {
  size_t index = GetIndex();
  size_t total_cases = active_programs * num_fixed;
  if (index >= total_cases) return;
  size_t fixed_idx = index / active_programs;
  size_t sample_idx = index % active_programs;
  uint8_t tape[2 * kSingleTapeSize] = {};
  const uint8_t *left = left_programs + sample_idx * kSingleTapeSize;
  const uint8_t *right = right_fixed + fixed_idx * kSingleTapeSize;
  for (size_t i = 0; i < kSingleTapeSize; ++i) {
    tape[i] = left[i];
    tape[i + kSingleTapeSize] = right[i];
  }
  Language::Evaluate(tape, kReplayStepCap, false);
  uint8_t *dst = right_offspring + index * kSingleTapeSize;
  for (size_t i = 0; i < kSingleTapeSize; ++i) {
    dst[i] = tape[i + kSingleTapeSize];
  }
}

template <typename Language>
__global__ void CheckSelfRepSubsetExact(uint8_t *programs, size_t seed,
                                        size_t seed_num_programs,
                                        const uint32_t *seed_indices,
                                        size_t active_programs, size_t *result,
                                        bool debug) {
  size_t index = GetIndex();
  constexpr size_t kNumIters = 13;
  constexpr size_t kNumExtraGens = 4;
  if (index >= active_programs) return;
  uint8_t tapes[kNumIters][2 * kSingleTapeSize] = {};
  uint64_t seed_index = seed_indices ? seed_indices[index] : index;
  uint64_t local_seed = SplitMix64(seed_num_programs * seed + seed_index);
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
    Language::Evaluate(tape, kReplayStepCap, eval_debug);

    for (size_t g = 0; g < kNumExtraGens; g++) {
      for (int j = 0; j < kSingleTapeSize; j++) {
        tape[j] = tape[j + kSingleTapeSize];
        tape[j + kSingleTapeSize] = noise[j];
      }
      Language::Evaluate(tape, kReplayStepCap, eval_debug);
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

void RewriteFixedSummary(const fs::path &path, const Config &cfg,
                         const std::vector<FixedStats> &stats) {
  FILE *f = CheckFopen(path.string().c_str(), "w");
  fprintf(f,
          "regime,seed,fixed_id,fixed_label,bracket_position,total_hits,"
          "total_tested,hit_fraction\n");
  for (const auto &row : stats) {
    double frac =
        row.total_tested ? static_cast<double>(row.total_hits) /
                               static_cast<double>(row.total_tested)
                         : 0.0;
    fprintf(f, "%s,%zu,%zu,\"%s\",%zu,%llu,%llu,%.12g\n", cfg.regime.c_str(),
            cfg.seed, row.fixed_id, CsvEscape(row.fixed_label).c_str(),
            row.bracket_position, (unsigned long long)row.total_hits,
            (unsigned long long)row.total_tested, frac);
  }
  fclose(f);
}

Outputs OpenOutputs(const Config &cfg) {
  RequireDirEmptyOrCreate(cfg.output_dir);
  Outputs out;
  out.fixed_summary_path = fs::path(cfg.output_dir) / "per_fixed_summary.csv";
  out.epoch_file = CheckFopen(
      (fs::path(cfg.output_dir) / "per_epoch_summary.csv").string().c_str(), "w");
  out.hit_file =
      CheckFopen((fs::path(cfg.output_dir) / "hit_cases.csv").string().c_str(), "w");
  out.log_file =
      CheckFopen((fs::path(cfg.output_dir) / "run_log.txt").string().c_str(), "w");
  fprintf(out.epoch_file,
          "regime,seed,epoch,fixed_id,hits_at_epoch,tested_at_epoch,"
          "hit_fraction_at_epoch\n");
  fprintf(out.hit_file,
          "regime,seed,epoch,soup_program_index,fixed_id,fixed_label,score\n");
  fflush(out.epoch_file);
  fflush(out.hit_file);
  return out;
}

void CloseOutputs(Outputs *outputs) {
  if (outputs->epoch_file) fclose(outputs->epoch_file);
  if (outputs->hit_file) fclose(outputs->hit_file);
  if (outputs->log_file) fclose(outputs->log_file);
}

void LogConfig(FILE *log_file, const Config &cfg,
               const std::vector<FixedTape> &fixed_tapes) {
  const char *visible = getenv("CUDA_VISIBLE_DEVICES");
  fprintf(log_file, "Pre-replicator availability experiment\n");
  fprintf(log_file, "regime=%s seed=%zu num_programs=%zu epochs=%zu\n",
          cfg.regime.c_str(), cfg.seed, cfg.num_programs, cfg.epochs);
  fprintf(log_file, "score_threshold=%zu\n", cfg.score_threshold);
  fprintf(log_file, "epoch_stride=%zu\n", cfg.epoch_stride);
  fprintf(log_file, "fixed_manifest=%s\n", cfg.fixed_manifest.c_str());
  fprintf(log_file, "fixed_side=right\n");
  fprintf(log_file, "score_target=adjusted_fixed_right_tape\n");
  fprintf(log_file, "N_mode=sparse_reinit_only\n");
  fprintf(log_file, "R_mode=dynamic_soup_snapshot\n");
  if (cfg.validation_program_limit > 0 || cfg.validation_fixed_limit > 0) {
    fprintf(log_file,
            "validation_mode=yes validation_program_limit=%zu "
            "validation_fixed_limit=%zu\n",
            cfg.validation_program_limit, cfg.validation_fixed_limit);
  } else {
    fprintf(log_file, "validation_mode=no\n");
  }
  if (cfg.gpu_label.has_value()) {
    fprintf(log_file, "requested_gpu=%s\n", cfg.gpu_label->c_str());
  }
  fprintf(log_file, "CUDA_VISIBLE_DEVICES=%s\n", visible ? visible : "(unset)");
#ifdef __CUDACC__
  int current_device = -1;
  cudaGetDevice(&current_device);
  fprintf(log_file, "visible_cuda_device_index=%d\n", current_device);
#endif
  fprintf(log_file, "fixed_tapes_loaded=%zu\n", fixed_tapes.size());
  for (const auto &fixed : fixed_tapes) {
    fprintf(log_file,
            "fixed_id=%zu label=%s bracket_position=%zu inert_byte=%u strip=%s "
            "hex=%s\n",
            fixed.fixed_id, fixed.fixed_label.c_str(), fixed.bracket_position,
            static_cast<unsigned int>(fixed.inert_byte), fixed.fixed_strip.c_str(),
            fixed.fixed_hex.c_str());
  }
  fprintf(log_file, "\n");
  fflush(log_file);
}

}  // namespace

int main(int argc, char **argv) {
  Config cfg = ParseArgs(argc, argv);
  auto fixed_tapes =
      LoadFixedManifest(cfg.fixed_manifest, cfg.validation_fixed_limit);
  std::vector<FixedStats> fixed_stats;
  fixed_stats.reserve(fixed_tapes.size());
  for (const auto &fixed : fixed_tapes) {
    fixed_stats.push_back(FixedStats{fixed.fixed_id, fixed.fixed_label,
                                     fixed.bracket_position, 0, 0});
  }

  Outputs outputs = OpenOutputs(cfg);
  LogConfig(outputs.log_file, cfg, fixed_tapes);

  const size_t active_programs =
      cfg.validation_program_limit > 0
          ? std::min(cfg.validation_program_limit, cfg.num_programs)
          : cfg.num_programs;
  const size_t batched_cases = active_programs * fixed_tapes.size();

  std::unique_ptr<DeviceMemory<uint8_t>> programs_dev;
  DeviceMemory<uint8_t> sampled_left_dev(active_programs * kSingleTapeSize);
  DeviceMemory<uint8_t> fixed_dev(fixed_tapes.size() * kSingleTapeSize);
  DeviceMemory<uint8_t> right_offspring_dev(batched_cases * kSingleTapeSize);
  DeviceMemory<size_t> scores_dev(batched_cases);
  DeviceMemory<uint32_t> score_seed_indices_dev(batched_cases);
  std::unique_ptr<DeviceMemory<uint32_t>> shuf_idx_dev;
  std::unique_ptr<DeviceMemory<unsigned long long>> insn_count;
  std::unique_ptr<DeviceMemory<unsigned long long>> step_count_sum;
  std::unique_ptr<DeviceMemory<unsigned long long>> termination_counts;

  std::vector<uint8_t> fixed_blob(fixed_tapes.size() * kSingleTapeSize);
  for (size_t i = 0; i < fixed_tapes.size(); ++i) {
    memcpy(&fixed_blob[i * kSingleTapeSize], fixed_tapes[i].bytes.data(),
           kSingleTapeSize);
  }
  fixed_dev.Write(fixed_blob.data(), fixed_blob.size());

  std::vector<uint32_t> score_seed_indices_host(batched_cases);
  for (size_t fixed_idx = 0; fixed_idx < fixed_tapes.size(); ++fixed_idx) {
    for (size_t sample_idx = 0; sample_idx < active_programs; ++sample_idx) {
      score_seed_indices_host[fixed_idx * active_programs + sample_idx] =
          static_cast<uint32_t>(sample_idx);
    }
  }
  score_seed_indices_dev.Write(score_seed_indices_host.data(), batched_cases);

  if (cfg.regime == "R") {
    programs_dev = std::make_unique<DeviceMemory<uint8_t>>(cfg.num_programs *
                                                           kSingleTapeSize);
    shuf_idx_dev = std::make_unique<DeviceMemory<uint32_t>>(cfg.num_programs);
    insn_count = std::make_unique<DeviceMemory<unsigned long long>>(1);
    step_count_sum = std::make_unique<DeviceMemory<unsigned long long>>(1);
    termination_counts =
        std::make_unique<DeviceMemory<unsigned long long>>(kNumTerminationReasons);
    const unsigned long long zero = 0;
    std::array<unsigned long long, kNumTerminationReasons>
        zero_termination_counts = {};
    insn_count->Write(&zero, 1);
    step_count_sum->Write(&zero, 1);
    termination_counts->Write(zero_termination_counts.data(),
                              zero_termination_counts.size());
    size_t grid = (cfg.num_programs + kThreads - 1) / kThreads;
    RUN(grid, kThreads, InitPrograms<Bff>, SimulationSeed(cfg.seed, 0),
        cfg.num_programs, programs_dev->Get(), false, nullptr);
    Synchronize();
  }

  std::vector<uint32_t> shuffle_indices(cfg.num_programs);
  std::vector<size_t> scores_host(batched_cases);
  const auto wall_start = std::chrono::steady_clock::now();

  for (size_t internal_epoch = 0; internal_epoch < cfg.epochs; ++internal_epoch) {
    const size_t logical_epoch = internal_epoch + 1;
    if (cfg.regime == "R") {
      BuildReplayShuffle(cfg.seed, internal_epoch, &shuffle_indices);
      shuf_idx_dev->Write(shuffle_indices.data(), shuffle_indices.size());
      size_t num_indices = cfg.num_programs / 2;
      size_t grid = (cfg.num_programs + 2 * kThreads - 1) / (2 * kThreads);
      RUN(grid, kThreads, MutateAndRunPrograms<Bff>, programs_dev->Get(),
          shuf_idx_dev->Get(), SimulationSeed(cfg.seed, internal_epoch), 0,
          insn_count->Get(), step_count_sum->Get(), termination_counts->Get(),
          cfg.num_programs, num_indices);
      Synchronize();
    }

    if (logical_epoch % cfg.epoch_stride != 0) {
      continue;
    }

    uint8_t *left_programs_dev = nullptr;
    if (cfg.regime == "R") {
      left_programs_dev = programs_dev->Get();
    } else {
      size_t grid = (active_programs + kThreads - 1) / kThreads;
      RUN(grid, kThreads, InitPrograms<Bff>,
          SimulationSeed(cfg.seed, logical_epoch), active_programs,
          sampled_left_dev.Get(), false, nullptr);
      Synchronize();
      left_programs_dev = sampled_left_dev.Get();
    }

    size_t batch_grid = (batched_cases + kThreads - 1) / kThreads;
    RUN(batch_grid, kThreads, ReplayFixedRightBatchKernel<Bff>, left_programs_dev,
        fixed_dev.Get(), right_offspring_dev.Get(), active_programs,
        fixed_tapes.size());
    RUN(batch_grid, kThreads, CheckSelfRepSubsetExact<Bff>,
        right_offspring_dev.Get(), PreReplicatorScoreSeed(cfg.seed, logical_epoch),
        active_programs, score_seed_indices_dev.Get(), batched_cases,
        scores_dev.Get(), false);
    Synchronize();
    scores_dev.Read(scores_host.data(), batched_cases);

    std::vector<size_t> hits_per_fixed(fixed_tapes.size(), 0);
    std::vector<uint32_t> hit_indices;
    for (size_t case_idx = 0; case_idx < batched_cases; ++case_idx) {
      if (scores_host[case_idx] >= cfg.score_threshold) {
        hit_indices.push_back(static_cast<uint32_t>(case_idx));
        hits_per_fixed[case_idx / active_programs]++;
      }
    }

    for (size_t fixed_idx = 0; fixed_idx < fixed_tapes.size(); ++fixed_idx) {
      size_t hits_at_epoch = hits_per_fixed[fixed_idx];
      fixed_stats[fixed_idx].total_hits += hits_at_epoch;
      fixed_stats[fixed_idx].total_tested += active_programs;
      double frac =
          active_programs ? static_cast<double>(hits_at_epoch) /
                                static_cast<double>(active_programs)
                          : 0.0;
      fprintf(outputs.epoch_file, "%s,%zu,%zu,%zu,%zu,%zu,%.12g\n",
              cfg.regime.c_str(), cfg.seed, logical_epoch,
              fixed_tapes[fixed_idx].fixed_id, hits_at_epoch, active_programs,
              frac);
    }
    fflush(outputs.epoch_file);

    if (!hit_indices.empty()) {
      for (uint32_t case_idx : hit_indices) {
        size_t fixed_idx = case_idx / active_programs;
        size_t sampled_pos = case_idx % active_programs;
        const auto &fixed = fixed_tapes[fixed_idx];
        fprintf(outputs.hit_file, "%s,%zu,%zu,%zu,%zu,\"%s\",%zu\n",
                cfg.regime.c_str(), cfg.seed, logical_epoch, sampled_pos,
                fixed.fixed_id, CsvEscape(fixed.fixed_label).c_str(),
                scores_host[case_idx]);
      }
      fflush(outputs.hit_file);
    }

    RewriteFixedSummary(outputs.fixed_summary_path, cfg, fixed_stats);
    const auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::steady_clock::now() - wall_start);
    fprintf(outputs.log_file,
            "processed_epoch=%zu elapsed_s=%lld active_programs=%zu\n",
            logical_epoch, static_cast<long long>(elapsed.count()),
            active_programs);
    fflush(outputs.log_file);
    fprintf(stdout, "[%s seed=%zu] processed epoch %zu/%zu\n",
            cfg.regime.c_str(), cfg.seed, logical_epoch, cfg.epochs);
    fflush(stdout);
  }

  const auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
      std::chrono::steady_clock::now() - wall_start);
  fprintf(outputs.log_file, "completed=yes elapsed_s=%lld\n",
          static_cast<long long>(elapsed.count()));
  fflush(outputs.log_file);
  CloseOutputs(&outputs);
  return 0;
}
