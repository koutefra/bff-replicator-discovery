#include <stdint.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <numeric>
#include <optional>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
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
constexpr size_t kDefaultEpochStride = 1;

struct PartnerTape {
  size_t partner_id = 0;
  std::string partner_group;
  std::string source_regime;
  size_t source_seed = 0;
  size_t source_discovery_epoch = 0;
  std::string partner_hex;
  std::string partner_strip;
  std::array<uint8_t, kSingleTapeSize> bytes = {};
};

struct Config {
  std::string regime;
  size_t seed = 0;
  size_t num_programs = kDefaultNumPrograms;
  size_t epochs = kDefaultEpochs;
  size_t score_threshold = kDefaultScoreThreshold;
  size_t epoch_stride = kDefaultEpochStride;
  size_t sample_programs = 0;
  size_t validation_program_limit = 0;
  size_t validation_partner_limit = 0;
  std::string partner_manifest;
  std::string output_dir;
  std::optional<std::string> checkpoint_dir;
  size_t checkpoint_interval = 0;
  std::optional<std::string> load_checkpoint;
  std::optional<std::string> gpu_label;
};

struct PartnerStats {
  size_t partner_id = 0;
  std::string partner_group;
  uint64_t total_hits = 0;
  uint64_t total_tested = 0;
};

struct Outputs {
  FILE *epoch_file = nullptr;
  FILE *hit_file = nullptr;
  FILE *log_file = nullptr;
  fs::path partner_summary_path;
};

bool FileExists(const fs::path &path) {
  std::error_code ec;
  return fs::exists(path, ec);
}

std::string BytesToHex(const uint8_t *data, size_t len) {
  static constexpr char kHexDigits[] = "0123456789abcdef";
  std::string out(len * 2, '0');
  for (size_t i = 0; i < len; ++i) {
    uint8_t byte = data[i];
    out[i * 2] = kHexDigits[byte >> 4];
    out[i * 2 + 1] = kHexDigits[byte & 0xF];
  }
  return out;
}

bool IsOpcodeByte(uint8_t byte) {
  switch (byte) {
    case '[':
    case ']':
    case '+':
    case '-':
    case '.':
    case ',':
    case '<':
    case '>':
    case '{':
    case '}':
      return true;
    default:
      return false;
  }
}

std::string OpcodeStripFromBytes(const uint8_t *data, size_t len) {
  std::string out;
  out.reserve(len);
  for (size_t i = 0; i < len; ++i) {
    if (IsOpcodeByte(data[i])) {
      out.push_back(static_cast<char>(data[i]));
    }
  }
  return out;
}

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

bool HasArg(int argc, char **argv, const std::string &key) {
  for (int i = 1; i < argc; ++i) {
    if (std::string(argv[i]) == key) {
      return true;
    }
  }
  return false;
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

void RequireDirExistsForResume(const fs::path &path) {
  if (!fs::exists(path) || !fs::is_directory(path)) {
    fprintf(stderr, "Resume output directory is missing: %s\n",
            path.string().c_str());
    exit(1);
  }
}

Config ParseArgs(int argc, char **argv) {
  Config cfg;
  auto regime = GetArgValue(argc, argv, "--regime");
  auto seed = GetArgValue(argc, argv, "--seed");
  auto output_dir = GetArgValue(argc, argv, "--output_dir");
  auto partner_manifest = GetArgValue(argc, argv, "--partner_manifest");
  if (!regime.has_value() || !seed.has_value() || !output_dir.has_value() ||
      !partner_manifest.has_value()) {
    fprintf(stderr,
            "Required args: --regime {N|R} --seed <int> --output_dir <dir> "
            "--partner_manifest <csv>\n");
    exit(1);
  }
  cfg.regime = *regime;
  if (cfg.regime != "N" && cfg.regime != "R") {
    fprintf(stderr, "--regime must be N or R\n");
    exit(1);
  }
  cfg.seed = strtoull(seed->c_str(), nullptr, 10);
  cfg.output_dir = *output_dir;
  cfg.partner_manifest = *partner_manifest;
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
  if (auto value = GetArgValue(argc, argv, "--sample_programs")) {
    cfg.sample_programs = strtoull(value->c_str(), nullptr, 10);
  }
  if (auto value = GetArgValue(argc, argv, "--validation_program_limit")) {
    cfg.validation_program_limit = strtoull(value->c_str(), nullptr, 10);
  }
  if (auto value = GetArgValue(argc, argv, "--validation_partner_limit")) {
    cfg.validation_partner_limit = strtoull(value->c_str(), nullptr, 10);
  }
  if (auto value = GetArgValue(argc, argv, "--checkpoint_dir")) {
    cfg.checkpoint_dir = *value;
  }
  if (auto value = GetArgValue(argc, argv, "--checkpoint_interval")) {
    cfg.checkpoint_interval = strtoull(value->c_str(), nullptr, 10);
  }
  if (auto value = GetArgValue(argc, argv, "--load_checkpoint")) {
    cfg.load_checkpoint = *value;
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

std::vector<PartnerTape> LoadPartnerManifest(const fs::path &path,
                                             size_t partner_limit) {
  FILE *f = CheckFopen(path.string().c_str(), "r");
  char *line = nullptr;
  size_t cap = 0;
  ssize_t line_len = getline(&line, &cap, f);
  if (line_len <= 0) {
    fprintf(stderr, "Empty partner manifest: %s\n", path.string().c_str());
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

  std::vector<PartnerTape> partners;
  while ((line_len = getline(&line, &cap, f)) > 0) {
    std::vector<std::string> row = ParseCsvLine(std::string(line, line_len));
    if (row.empty() || row[0].empty()) {
      continue;
    }
    PartnerTape partner;
    partner.partner_id = strtoull(field(row, "partner_id").c_str(), nullptr, 10);
    partner.partner_group = field(row, "partner_group");
    partner.source_regime = field(row, "source_regime");
    partner.source_seed = strtoull(field(row, "source_seed").c_str(), nullptr, 10);
    partner.source_discovery_epoch =
        strtoull(field(row, "source_discovery_epoch").c_str(), nullptr, 10);
    partner.partner_hex = field(row, "partner_hex");
    partner.partner_strip = field(row, "partner_strip");
    partner.bytes = HexToTape(partner.partner_hex);
    partners.push_back(partner);
    if (partner_limit > 0 && partners.size() >= partner_limit) {
      break;
    }
  }
  free(line);
  fclose(f);
  if (partners.empty()) {
    fprintf(stderr, "No partners loaded from %s\n", path.string().c_str());
    exit(1);
  }
  return partners;
}

size_t ReplaySelfrepSeed(size_t run_seed, size_t logical_epoch) {
  return SplitMix64(SplitMix64(run_seed) ^ SplitMix64(logical_epoch - 1));
}

template <typename Language>
__global__ void ReplayFixedLeftPartnerKernel(const uint8_t *right_programs,
                                             const uint8_t *left_partner,
                                             uint8_t *right_offspring,
                                             size_t active_programs) {
  size_t index = GetIndex();
  if (index >= active_programs) return;
  uint8_t tape[2 * kSingleTapeSize] = {};
  const uint8_t *right = right_programs + index * kSingleTapeSize;
  for (size_t i = 0; i < kSingleTapeSize; ++i) {
    tape[i] = left_partner[i];
    tape[i + kSingleTapeSize] = right[i];
  }
  Language::Evaluate(tape, kReplayStepCap, false);
  uint8_t *dst = right_offspring + index * kSingleTapeSize;
  for (size_t i = 0; i < kSingleTapeSize; ++i) {
    dst[i] = tape[i + kSingleTapeSize];
  }
}

template <typename Language>
__global__ void ReplayFixedLeftPartnerBatchKernel(const uint8_t *right_programs,
                                                  const uint8_t *left_partners,
                                                  uint8_t *right_offspring,
                                                  size_t active_programs,
                                                  size_t num_partners) {
  size_t index = GetIndex();
  size_t total_cases = active_programs * num_partners;
  if (index >= total_cases) return;
  size_t partner_idx = index / active_programs;
  size_t sample_idx = index % active_programs;
  uint8_t tape[2 * kSingleTapeSize] = {};
  const uint8_t *left = left_partners + partner_idx * kSingleTapeSize;
  const uint8_t *right = right_programs + sample_idx * kSingleTapeSize;
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

enum StaticFamilyClass : uint8_t {
  kFamilyNone = 0,
  kFamilyNine = 1,
  kFamilyTen = 2,
};

__device__ __host__ bool IsTrackedOp(uint8_t b) {
  return b == 0 || b == '[' || b == ']' || b == '+' || b == '-' || b == '.' ||
         b == ',' || b == '<' || b == '>' || b == '{' || b == '}';
}

__device__ __host__ char OpSymbol(uint8_t b) {
  return b == 0 ? '0' : static_cast<char>(b);
}

__device__ __host__ bool IsCopyOp(char c) { return c == '.' || c == ','; }
__device__ __host__ bool IsH0Op(char c) { return c == '<' || c == '>'; }
__device__ __host__ bool IsH1Op(char c) { return c == '{' || c == '}'; }
__device__ __host__ bool IsBracketOp(char c) { return c == '[' || c == ']'; }

__device__ __host__ bool IsCancelPair(char a, char b) {
  return (a == '>' && b == '<') || (a == '<' && b == '>') ||
         (a == '}' && b == '{') || (a == '{' && b == '}');
}

__device__ __host__ int CleanseOps(const uint8_t *tape, char *ops) {
  int len = 0;
  for (size_t i = 0; i < kSingleTapeSize; ++i) {
    uint8_t b = tape[i];
    if (!IsTrackedOp(b)) {
      continue;
    }
    char op = OpSymbol(b);
    if (op == '0') {
      continue;
    }
    if (len > 0) {
      char last = ops[len - 1];
      if (IsCopyOp(op) && IsCopyOp(last)) {
        continue;
      }
      if (IsCancelPair(last, op)) {
        len--;
        continue;
      }
    }
    ops[len++] = op;
  }
  return len;
}

__device__ __host__ bool IsValidReplicatorLoop(const char *ops, int start,
                                               int end) {
  bool has_copy = false;
  int h0 = 0;
  int h1 = 0;
  for (int i = start; i < end; ++i) {
    char op = ops[i];
    if (IsCopyOp(op)) {
      has_copy = true;
    } else if (op == '<') {
      h0--;
    } else if (op == '>') {
      h0++;
    } else if (op == '{') {
      h1--;
    } else if (op == '}') {
      h1++;
    }
  }
  if (!has_copy) {
    return false;
  }
  if ((h0 != -2 && h0 != -1 && h0 != 1 && h0 != 2) ||
      (h1 != -2 && h1 != -1 && h1 != 1 && h1 != 2)) {
    return false;
  }
  return h0 == -h1;
}

__device__ __host__ bool FindReplicatorBody(const char *ops, int n, int *start,
                                            int *end) {
  for (int i = 0; i < n; ++i) {
    if (ops[i] != '[') {
      continue;
    }
    int j = i + 1;
    while (j < n && !IsBracketOp(ops[j])) {
      j++;
    }
    if (j < n && ops[j] == ']') {
      if (IsValidReplicatorLoop(ops, i + 1, j)) {
        *start = i;
        *end = j + 1;
        return true;
      }
    }
  }
  return false;
}

__device__ __host__ uint8_t ClassifyNineTenTape(const uint8_t *tape) {
  char ops[kSingleTapeSize];
  char reversed_ops[kSingleTapeSize];
  int n = CleanseOps(tape, ops);
  int body1_start = 0;
  int body1_end = 0;
  if (!FindReplicatorBody(ops, n, &body1_start, &body1_end)) {
    return kFamilyNone;
  }
  for (int i = 0; i < n; ++i) {
    reversed_ops[i] = ops[n - 1 - i];
  }
  int body2r_start = 0;
  int body2r_end = 0;
  if (!FindReplicatorBody(reversed_ops, n, &body2r_start, &body2r_end)) {
    return kFamilyNone;
  }
  int body2_start = n - body2r_end;
  return body1_end == body2_start ? kFamilyNine : kFamilyTen;
}

__global__ void ClassifyNineTenStaticKernel(const uint8_t *programs,
                                            uint8_t *result, size_t count) {
  size_t index = GetIndex();
  if (index >= count) return;
  result[index] = ClassifyNineTenTape(programs + index * kSingleTapeSize);
}

const char *FamilyClassName(uint8_t family_class) {
  switch (family_class) {
    case kFamilyNine:
      return "nine";
    case kFamilyTen:
      return "ten";
    default:
      return "none";
  }
}

// Same scoring logic as common_language.h::CheckSelfRep, but with separate
// active_programs and seed_num_programs so subset validation can safely reuse
// the original seeding convention from the full soup size.
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

__global__ void GatherProgramsKernel(const uint8_t *programs,
                                     const uint32_t *indices,
                                     uint8_t *gathered, size_t count) {
  size_t index = GetIndex();
  if (index >= count) return;
  const uint8_t *src = programs + indices[index] * kSingleTapeSize;
  uint8_t *dst = gathered + index * kSingleTapeSize;
  for (size_t i = 0; i < kSingleTapeSize; ++i) {
    dst[i] = src[i];
  }
}

uint64_t ReplaySamplingSeed(size_t run_seed, size_t logical_epoch) {
  return SplitMix64(SplitMix64(run_seed) ^ SplitMix64(0x9e3779b97f4a7c15ULL) ^
                    SplitMix64(logical_epoch));
}

void SampleProgramIndices(size_t total_programs, size_t sample_programs,
                          uint64_t seed, std::vector<uint32_t> *indices) {
  CHECK(sample_programs <= total_programs);
  indices->resize(total_programs);
  std::iota(indices->begin(), indices->end(), 0u);
  for (size_t i = 0; i < sample_programs; ++i) {
    uint64_t step_seed = SplitMix64(seed ^ SplitMix64(i + 1));
    size_t j = i + (step_seed % (total_programs - i));
    std::swap((*indices)[i], (*indices)[j]);
  }
  indices->resize(sample_programs);
}

uint64_t SimulationSeed(size_t run_seed, size_t seed2) {
  return SplitMix64(SplitMix64(run_seed) ^ SplitMix64(seed2));
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

size_t LoadReplayCheckpoint(const std::string &path, size_t expected_num_programs,
                            DeviceMemory<uint8_t> *programs,
                            std::vector<uint8_t> *host_buffer) {
  FILE *f = CheckFopen(path.c_str(), "r");
  size_t reset_index = 0;
  size_t file_num_programs = 0;
  size_t next_internal_epoch = 0;
  CHECK(fread(&reset_index, sizeof(reset_index), 1, f) == 1);
  CHECK(fread(&file_num_programs, sizeof(file_num_programs), 1, f) == 1);
  CHECK(fread(&next_internal_epoch, sizeof(next_internal_epoch), 1, f) == 1);
  CHECK(file_num_programs == expected_num_programs);
  host_buffer->resize(expected_num_programs * kSingleTapeSize);
  CHECK(fread(host_buffer->data(), 1, host_buffer->size(), f) ==
        host_buffer->size());
  fclose(f);
  programs->Write(host_buffer->data(), host_buffer->size());
  return next_internal_epoch;
}

void SaveReplayCheckpoint(const std::string &dir, size_t completed_epoch,
                          size_t next_internal_epoch, size_t num_programs,
                          DeviceMemory<uint8_t> *programs,
                          std::vector<uint8_t> *host_buffer) {
  CHECK(fs::create_directories(dir) || fs::exists(dir));
  host_buffer->resize(num_programs * kSingleTapeSize);
  programs->Read(host_buffer->data(), host_buffer->size());
  std::vector<char> save_path(dir.size() + 20);
  snprintf(save_path.data(), save_path.size(), "%s/%010zu.dat", dir.c_str(),
           completed_epoch);
  FILE *f = CheckFopen(save_path.data(), "w");
  size_t reset_index = 1;
  fwrite(&reset_index, sizeof(reset_index), 1, f);
  fwrite(&num_programs, sizeof(num_programs), 1, f);
  fwrite(&next_internal_epoch, sizeof(next_internal_epoch), 1, f);
  fwrite(host_buffer->data(), 1, host_buffer->size(), f);
  fclose(f);
}

void RewritePartnerSummary(const fs::path &path, const Config &cfg,
                           const std::vector<PartnerStats> &stats) {
  FILE *f = CheckFopen(path.string().c_str(), "w");
  fprintf(f,
          "regime,seed,partner_id,partner_group,total_hits,total_tested,"
          "hit_fraction\n");
  for (const auto &row : stats) {
    double frac =
        row.total_tested ? (double)row.total_hits / (double)row.total_tested : 0.0;
    fprintf(f, "%s,%zu,%zu,\"%s\",%llu,%llu,%.12g\n", cfg.regime.c_str(),
            cfg.seed, row.partner_id, CsvEscape(row.partner_group).c_str(),
            (unsigned long long)row.total_hits,
            (unsigned long long)row.total_tested, frac);
  }
  fclose(f);
}

void RestorePartnerStatsFromEpochSummary(const fs::path &path, const Config &cfg,
                                         std::vector<PartnerStats> *stats) {
  if (!FileExists(path)) {
    return;
  }
  std::ifstream in(path);
  CHECK(in.good());
  std::string line;
  if (!std::getline(in, line)) {
    return;
  }
  std::unordered_map<size_t, size_t> partner_to_idx;
  for (size_t i = 0; i < stats->size(); ++i) {
    partner_to_idx[(*stats)[i].partner_id] = i;
  }
  while (std::getline(in, line)) {
    if (line.empty()) {
      continue;
    }
    std::stringstream ss(line);
    std::string regime;
    std::string seed_str;
    std::string epoch_str;
    std::string partner_id_str;
    std::string hits_str;
    std::string tested_str;
    std::string frac_str;
    CHECK(std::getline(ss, regime, ','));
    CHECK(std::getline(ss, seed_str, ','));
    CHECK(std::getline(ss, epoch_str, ','));
    CHECK(std::getline(ss, partner_id_str, ','));
    CHECK(std::getline(ss, hits_str, ','));
    CHECK(std::getline(ss, tested_str, ','));
    CHECK(std::getline(ss, frac_str, ','));
    if (regime != cfg.regime) {
      continue;
    }
    if (strtoull(seed_str.c_str(), nullptr, 10) != cfg.seed) {
      continue;
    }
    size_t partner_id = strtoull(partner_id_str.c_str(), nullptr, 10);
    auto it = partner_to_idx.find(partner_id);
    CHECK(it != partner_to_idx.end());
    (*stats)[it->second].total_hits += strtoull(hits_str.c_str(), nullptr, 10);
    (*stats)[it->second].total_tested +=
        strtoull(tested_str.c_str(), nullptr, 10);
  }
}

Outputs OpenOutputs(const Config &cfg, bool append_existing) {
  if (append_existing) {
    RequireDirExistsForResume(cfg.output_dir);
  } else {
    RequireDirEmptyOrCreate(cfg.output_dir);
  }
  Outputs out;
  out.partner_summary_path = fs::path(cfg.output_dir) / "per_partner_summary.csv";
  const char *mode = append_existing ? "a" : "w";
  out.epoch_file =
      CheckFopen((fs::path(cfg.output_dir) / "per_epoch_summary.csv").string().c_str(), mode);
  out.hit_file =
      CheckFopen((fs::path(cfg.output_dir) / "hit_cases.csv").string().c_str(), mode);
  out.log_file =
      CheckFopen((fs::path(cfg.output_dir) / "run_log.txt").string().c_str(), mode);
  if (!append_existing) {
    fprintf(out.epoch_file,
            "regime,seed,epoch,partner_id,hits_at_epoch,tested_at_epoch,"
            "hit_fraction_at_epoch\n");
    fprintf(out.hit_file,
            "regime,seed,epoch,soup_program_index,partner_id,partner_group,score,"
            "right_offspring_tape_raw_bytes,right_offspring_tape_opcode_strip,"
            "original_right_tape_raw_bytes,original_right_tape_opcode_strip\n");
  }
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
               const std::vector<PartnerTape> &partners) {
  const char *visible = getenv("CUDA_VISIBLE_DEVICES");
  fprintf(log_file, "Fixed-partner replay experiment\n");
  fprintf(log_file, "regime=%s seed=%zu num_programs=%zu epochs=%zu\n",
          cfg.regime.c_str(), cfg.seed, cfg.num_programs, cfg.epochs);
  fprintf(log_file, "score_threshold=%zu\n", cfg.score_threshold);
  fprintf(log_file, "epoch_stride=%zu\n", cfg.epoch_stride);
  fprintf(log_file, "sample_programs=%zu\n", cfg.sample_programs);
  fprintf(log_file, "partner_manifest=%s\n", cfg.partner_manifest.c_str());
  fprintf(log_file, "exact_partner_recovery=yes\n");
  fprintf(log_file, "partners_loaded=%zu\n", partners.size());
  if (cfg.validation_program_limit > 0 || cfg.validation_partner_limit > 0) {
    fprintf(log_file,
            "validation_mode=yes validation_program_limit=%zu "
            "validation_partner_limit=%zu\n",
            cfg.validation_program_limit, cfg.validation_partner_limit);
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
  if (cfg.checkpoint_dir.has_value()) {
    fprintf(log_file, "checkpoint_dir=%s checkpoint_interval=%zu\n",
            cfg.checkpoint_dir->c_str(), cfg.checkpoint_interval);
  } else {
    fprintf(log_file, "checkpoint_dir=(disabled)\n");
  }
  if (cfg.load_checkpoint.has_value()) {
    fprintf(log_file, "load_checkpoint=%s\n", cfg.load_checkpoint->c_str());
  }
  fprintf(log_file, "\nLoaded partners:\n");
  for (const auto &partner : partners) {
    fprintf(log_file,
            "partner_id=%zu group=%s source=%s seed=%zu discovery_epoch=%zu "
            "strip=%s\n",
            partner.partner_id, partner.partner_group.c_str(),
            partner.source_regime.c_str(), partner.source_seed,
            partner.source_discovery_epoch, partner.partner_strip.c_str());
  }
  fprintf(log_file, "\n");
  fflush(log_file);
}

}  // namespace

int main(int argc, char **argv) {
  Config cfg = ParseArgs(argc, argv);
  auto partners = LoadPartnerManifest(cfg.partner_manifest,
                                      cfg.validation_partner_limit);
  std::vector<PartnerStats> partner_stats;
  partner_stats.reserve(partners.size());
  for (const auto &partner : partners) {
    partner_stats.push_back(
        PartnerStats{partner.partner_id, partner.partner_group, 0, 0});
  }
  const fs::path epoch_summary_path =
      fs::path(cfg.output_dir) / "per_epoch_summary.csv";
  const bool append_existing =
      cfg.load_checkpoint.has_value() && FileExists(epoch_summary_path);
  if (append_existing) {
    RestorePartnerStatsFromEpochSummary(epoch_summary_path, cfg, &partner_stats);
  }

  Outputs outputs = OpenOutputs(cfg, append_existing);
  LogConfig(outputs.log_file, cfg, partners);
  fprintf(outputs.log_file, "resume_outputs=%s\n",
          append_existing ? "yes" : "no");
  fflush(outputs.log_file);
  if (append_existing) {
    RewritePartnerSummary(outputs.partner_summary_path, cfg, partner_stats);
  }

  const size_t active_programs =
      cfg.validation_program_limit > 0
          ? std::min(cfg.validation_program_limit, cfg.num_programs)
          : (cfg.sample_programs > 0
                 ? std::min(cfg.sample_programs, cfg.num_programs)
                 : cfg.num_programs);
  const size_t batched_cases = active_programs * partners.size();

  DeviceMemory<uint8_t> programs_dev(cfg.num_programs * kSingleTapeSize);
  DeviceMemory<uint8_t> soup_dev(active_programs * kSingleTapeSize);
  DeviceMemory<uint8_t> offspring_dev(batched_cases * kSingleTapeSize);
  DeviceMemory<size_t> scores_dev(batched_cases);
  DeviceMemory<uint32_t> sample_indices_dev(active_programs);
  DeviceMemory<uint32_t> score_seed_indices_dev(batched_cases);
  DeviceMemory<uint32_t> shuf_idx_dev(cfg.num_programs);
  DeviceMemory<uint8_t> partner_dev(partners.size() * kSingleTapeSize);
  DeviceMemory<unsigned long long> insn_count(1);
  DeviceMemory<unsigned long long> step_count_sum(1);
  DeviceMemory<unsigned long long> termination_counts(kNumTerminationReasons);
  std::vector<uint8_t> partner_blob(partners.size() * kSingleTapeSize);
  for (size_t i = 0; i < partners.size(); ++i) {
    memcpy(&partner_blob[i * kSingleTapeSize], partners[i].bytes.data(),
           kSingleTapeSize);
  }
  partner_dev.Write(partner_blob.data(), partner_blob.size());
  unsigned long long zero = 0;
  std::array<unsigned long long, kNumTerminationReasons> zero_termination_counts = {};
  insn_count.Write(&zero, 1);
  step_count_sum.Write(&zero, 1);
  termination_counts.Write(zero_termination_counts.data(),
                           zero_termination_counts.size());

  std::vector<size_t> scores_host(batched_cases);
  std::vector<uint32_t> hit_indices;
  std::vector<uint8_t> gathered_offspring_host;
  std::vector<uint8_t> sampled_original_host;
  std::vector<uint32_t> sampled_indices_host;
  std::vector<uint32_t> sample_pool_indices;
  std::vector<uint32_t> batched_seed_indices_host(batched_cases);
  std::vector<uint32_t> shuffle_indices(cfg.num_programs);
  std::vector<uint8_t> checkpoint_host;

  size_t start_internal_epoch = 0;
  if (cfg.load_checkpoint.has_value()) {
    start_internal_epoch = LoadReplayCheckpoint(*cfg.load_checkpoint,
                                                cfg.num_programs,
                                                &programs_dev, &checkpoint_host);
  } else {
    size_t grid = (cfg.num_programs + kThreads - 1) / kThreads;
    RUN(grid, kThreads, InitPrograms<Bff>, SimulationSeed(cfg.seed, 0),
        cfg.num_programs, programs_dev.Get(), false, nullptr);
    Synchronize();
  }

  const auto wall_start = std::chrono::steady_clock::now();
  bool completed = start_internal_epoch >= cfg.epochs;
  for (size_t internal_epoch = start_internal_epoch; internal_epoch < cfg.epochs;
       ++internal_epoch) {
    if (cfg.regime == "N") {
      size_t grid = (cfg.num_programs + kThreads - 1) / kThreads;
      RUN(grid, kThreads, MutateAndRunProgramsRandomPartner<Bff>,
          programs_dev.Get(), cfg.seed, internal_epoch, 0, insn_count.Get(),
          step_count_sum.Get(), termination_counts.Get(), cfg.num_programs);
    } else {
      BuildReplayShuffle(cfg.seed, internal_epoch, &shuffle_indices);
      shuf_idx_dev.Write(shuffle_indices.data(), shuffle_indices.size());
      size_t num_indices = cfg.num_programs / 2;
      size_t grid = (cfg.num_programs + 2 * kThreads - 1) / (2 * kThreads);
      RUN(grid, kThreads, MutateAndRunPrograms<Bff>, programs_dev.Get(),
          shuf_idx_dev.Get(), SimulationSeed(cfg.seed, internal_epoch), 0,
          insn_count.Get(), step_count_sum.Get(), termination_counts.Get(),
          cfg.num_programs, num_indices);
    }
    Synchronize();

    size_t logical_epoch = internal_epoch + 1;
    bool process_epoch = (logical_epoch % cfg.epoch_stride == 0);
    if (process_epoch) {
      sampled_indices_host.clear();
      if (cfg.validation_program_limit > 0) {
        sampled_indices_host.reserve(active_programs);
        for (size_t i = 0; i < active_programs; ++i) {
          sampled_indices_host.push_back(i);
        }
      } else if (cfg.sample_programs > 0) {
        SampleProgramIndices(cfg.num_programs, active_programs,
                             ReplaySamplingSeed(cfg.seed, logical_epoch),
                             &sample_pool_indices);
        sampled_indices_host = sample_pool_indices;
      } else {
        sampled_indices_host.resize(active_programs);
        for (size_t i = 0; i < active_programs; ++i) {
          sampled_indices_host[i] = i;
        }
      }

      sample_indices_dev.Write(sampled_indices_host.data(), active_programs);
      size_t sample_grid = (active_programs + kThreads - 1) / kThreads;
      RUN(sample_grid, kThreads, GatherProgramsKernel, programs_dev.Get(),
          sample_indices_dev.Get(), soup_dev.Get(), active_programs);
      Synchronize();
      for (size_t partner_idx = 0; partner_idx < partners.size(); ++partner_idx) {
        for (size_t sample_idx = 0; sample_idx < active_programs; ++sample_idx) {
          batched_seed_indices_host[partner_idx * active_programs + sample_idx] =
              sampled_indices_host[sample_idx];
        }
      }
      score_seed_indices_dev.Write(batched_seed_indices_host.data(), batched_cases);
      size_t score_seed = ReplaySelfrepSeed(cfg.seed, logical_epoch);
      size_t batch_grid = (batched_cases + kThreads - 1) / kThreads;
      RUN(batch_grid, kThreads, ReplayFixedLeftPartnerBatchKernel<Bff>,
          soup_dev.Get(), partner_dev.Get(), offspring_dev.Get(), active_programs,
          partners.size());
      RUN(batch_grid, kThreads, CheckSelfRepSubsetExact<Bff>, offspring_dev.Get(),
          score_seed, cfg.num_programs, score_seed_indices_dev.Get(),
          batched_cases, scores_dev.Get(), false);
      Synchronize();
      scores_dev.Read(scores_host.data(), batched_cases);

      std::vector<size_t> hits_per_partner(partners.size(), 0);
      hit_indices.clear();
      for (size_t case_idx = 0; case_idx < batched_cases; ++case_idx) {
        if (scores_host[case_idx] >= cfg.score_threshold) {
          hit_indices.push_back(static_cast<uint32_t>(case_idx));
          hits_per_partner[case_idx / active_programs]++;
        }
      }

      for (size_t partner_idx = 0; partner_idx < partners.size(); ++partner_idx) {
        const auto &partner = partners[partner_idx];
        size_t hits_at_epoch = hits_per_partner[partner_idx];
        partner_stats[partner_idx].total_hits += hits_at_epoch;
        partner_stats[partner_idx].total_tested += active_programs;
        double frac =
            active_programs ? (double)hits_at_epoch / (double)active_programs : 0.0;
        fprintf(outputs.epoch_file, "%s,%zu,%zu,%zu,%zu,%zu,%.12g\n",
                cfg.regime.c_str(), cfg.seed, logical_epoch, partner.partner_id,
                hits_at_epoch, active_programs, frac);
      }
      fflush(outputs.epoch_file);

      if (!hit_indices.empty()) {
        sampled_original_host.resize(active_programs * kSingleTapeSize);
        gathered_offspring_host.resize(batched_cases * kSingleTapeSize);
        soup_dev.Read(sampled_original_host.data(), sampled_original_host.size());
        offspring_dev.Read(gathered_offspring_host.data(),
                           gathered_offspring_host.size());
        for (size_t hit_pos = 0; hit_pos < hit_indices.size(); ++hit_pos) {
          size_t case_idx = hit_indices[hit_pos];
          size_t partner_idx = case_idx / active_programs;
          size_t sampled_pos = case_idx % active_programs;
          size_t soup_index = sampled_indices_host[sampled_pos];
          const auto &partner = partners[partner_idx];
          const uint8_t *offspring =
              gathered_offspring_host.data() + case_idx * kSingleTapeSize;
          const uint8_t *original =
              sampled_original_host.data() + sampled_pos * kSingleTapeSize;
          std::string offspring_hex = BytesToHex(offspring, kSingleTapeSize);
          std::string offspring_strip =
              OpcodeStripFromBytes(offspring, kSingleTapeSize);
          std::string original_hex = BytesToHex(original, kSingleTapeSize);
          std::string original_strip =
              OpcodeStripFromBytes(original, kSingleTapeSize);
          fprintf(outputs.hit_file,
                  "%s,%zu,%zu,%zu,%zu,\"%s\",%zu,\"%s\",\"%s\",\"%s\",\"%s\"\n",
                  cfg.regime.c_str(), cfg.seed, logical_epoch, soup_index,
                  partner.partner_id, CsvEscape(partner.partner_group).c_str(),
                  scores_host[case_idx], offspring_hex.c_str(),
                  CsvEscape(offspring_strip).c_str(), original_hex.c_str(),
                  CsvEscape(original_strip).c_str());
        }
        fflush(outputs.hit_file);
      }

      RewritePartnerSummary(outputs.partner_summary_path, cfg, partner_stats);
      const auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
          std::chrono::steady_clock::now() - wall_start);
      fprintf(outputs.log_file,
              "processed_epoch=%zu elapsed_s=%lld active_programs=%zu\n",
              logical_epoch, (long long)elapsed.count(), active_programs);
      fflush(outputs.log_file);
      fprintf(stdout, "[%s seed=%zu] processed epoch %zu/%zu\n",
              cfg.regime.c_str(), cfg.seed, logical_epoch, cfg.epochs);
      fflush(stdout);
    }

    if (cfg.checkpoint_dir.has_value() && cfg.checkpoint_interval > 0 &&
        logical_epoch % cfg.checkpoint_interval == 0) {
      SaveReplayCheckpoint(*cfg.checkpoint_dir, logical_epoch, logical_epoch,
                           cfg.num_programs, &programs_dev, &checkpoint_host);
    }
  }
  completed = true;

  const auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
      std::chrono::steady_clock::now() - wall_start);
  fprintf(outputs.log_file, "completed=%s elapsed_s=%lld\n",
          completed ? "yes" : "no", (long long)elapsed.count());
  fflush(outputs.log_file);
  CloseOutputs(&outputs);
  return 0;
}
