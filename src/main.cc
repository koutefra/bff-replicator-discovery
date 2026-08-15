// Copyright 2024 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <assert.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <errno.h>
#include <optional>
#include <string>
#include <type_traits>
#include <vector>

#include "common.h"

namespace {

std::string BytesToHex(const uint8_t *data, size_t len) {
  static constexpr char kHexDigits[] = "0123456789abcdef";
  std::string out(len * 2, '0');
  for (size_t i = 0; i < len; ++i) {
    uint8_t byte = data[i];
    out[i * 2] = kHexDigits[byte >> 4];
    out[i * 2 + 1] = kHexDigits[byte & 0xf];
  }
  return out;
}

std::string JsonArray(const std::vector<std::string> &values) {
  std::string out;
  out.push_back('[');
  for (size_t i = 0; i < values.size(); ++i) {
    if (i > 0) {
      out.push_back(',');
    }
    out.push_back('"');
    out.append(values[i]);
    out.push_back('"');
  }
  out.push_back(']');
  return out;
}

std::string JsonArray(const std::vector<size_t> &values) {
  std::string out;
  out.push_back('[');
  for (size_t i = 0; i < values.size(); ++i) {
    if (i > 0) {
      out.push_back(',');
    }
    out.append(std::to_string(values[i]));
  }
  out.push_back(']');
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

std::string TrimAsciiWhitespace(std::string value) {
  size_t start = 0;
  while (start < value.size() &&
         std::isspace(static_cast<unsigned char>(value[start]))) {
    start++;
  }
  size_t end = value.size();
  while (end > start &&
         std::isspace(static_cast<unsigned char>(value[end - 1]))) {
    end--;
  }
  return value.substr(start, end - start);
}

std::vector<std::string> SplitCommaSeparatedPaths(const std::string &value) {
  std::vector<std::string> paths;
  size_t start = 0;
  while (start <= value.size()) {
    size_t comma = value.find(',', start);
    size_t end = (comma == std::string::npos) ? value.size() : comma;
    std::string token = TrimAsciiWhitespace(value.substr(start, end - start));
    if (!token.empty()) {
      paths.push_back(token);
    }
    if (comma == std::string::npos) {
      break;
    }
    start = comma + 1;
  }
  return paths;
}

constexpr char kInitSymbolProbOrder[] = "[]+-.,<>{}0";
constexpr size_t kNumInitSymbols = 11;
constexpr size_t kNumUnspecifiedInitBytes = 256 - kNumInitSymbols;
constexpr uint64_t kInitProbScale = uint64_t{1} << 63;

bool BuildInitByteCdf(const std::string &spec, std::vector<uint64_t> *cdf_out,
                      std::string *error_out) {
  std::vector<std::string> tokens;
  size_t start = 0;
  while (start <= spec.size()) {
    size_t comma = spec.find(',', start);
    size_t end = (comma == std::string::npos) ? spec.size() : comma;
    tokens.push_back(TrimAsciiWhitespace(spec.substr(start, end - start)));
    if (comma == std::string::npos) {
      break;
    }
    start = comma + 1;
  }

  if (tokens.size() != kNumInitSymbols) {
    *error_out =
        "expected 11 comma-separated probabilities in fixed order []+-.,<>{}0";
    return false;
  }

  std::vector<double> symbol_probs(kNumInitSymbols, 0.0);
  double specified_sum = 0.0;
  for (size_t i = 0; i < tokens.size(); ++i) {
    if (tokens[i].empty()) {
      *error_out = "empty probability entry in --init_symbol_probs";
      return false;
    }
    size_t parsed_chars = 0;
    double prob = 0.0;
    try {
      prob = std::stod(tokens[i], &parsed_chars);
    } catch (const std::exception &) {
      *error_out = "failed to parse probability `" + tokens[i] + "`";
      return false;
    }
    if (parsed_chars != tokens[i].size()) {
      *error_out = "invalid probability token `" + tokens[i] + "`";
      return false;
    }
    if (!std::isfinite(prob) || prob < 0.0) {
      *error_out = "probabilities must be finite and >= 0";
      return false;
    }
    symbol_probs[i] = prob;
    specified_sum += prob;
  }

  if (specified_sum > 1.0 + 1e-12) {
    *error_out = "symbol probabilities sum to more than 1";
    return false;
  }

  double remaining_mass = std::max(0.0, 1.0 - specified_sum);
  double fallback_prob =
      remaining_mass / static_cast<double>(kNumUnspecifiedInitBytes);
  std::vector<double> byte_probs(256, fallback_prob);
  const uint8_t symbol_bytes[kNumInitSymbols] = {
      static_cast<uint8_t>('['), static_cast<uint8_t>(']'),
      static_cast<uint8_t>('+'), static_cast<uint8_t>('-'),
      static_cast<uint8_t>('.'), static_cast<uint8_t>(','),
      static_cast<uint8_t>('<'), static_cast<uint8_t>('>'),
      static_cast<uint8_t>('{'), static_cast<uint8_t>('}'),
      static_cast<uint8_t>(0),
  };
  for (size_t i = 0; i < kNumInitSymbols; ++i) {
    byte_probs[symbol_bytes[i]] = symbol_probs[i];
  }

  cdf_out->assign(256, 0);
  long double cumulative = 0.0L;
  for (size_t byte = 0; byte < 256; ++byte) {
    cumulative += static_cast<long double>(byte_probs[byte]);
    cumulative = std::clamp(cumulative, 0.0L, 1.0L);
    long double scaled =
        cumulative * static_cast<long double>(kInitProbScale);
    uint64_t threshold = static_cast<uint64_t>(std::floor(scaled));
    if (threshold > kInitProbScale) {
      threshold = kInitProbScale;
    }
    (*cdf_out)[byte] = threshold;
  }
  (*cdf_out)[255] = kInitProbScale;
  return true;
}

bool LoadSnapshotSoup(const std::string &path, std::vector<uint8_t> *soup_out,
                      size_t *num_programs_out) {
  FILE *f = fopen(path.c_str(), "r");
  if (!f) {
    char buf[4096];
    fprintf(stderr, "Could not open %s: %s\n", path.c_str(),
            strerror_r(errno, buf, sizeof(buf)));
    return false;
  }

  size_t reset_index = 0;
  size_t num_programs = 0;
  size_t epoch = 0;
  if (fread(&reset_index, sizeof(reset_index), 1, f) != 1 ||
      fread(&num_programs, sizeof(num_programs), 1, f) != 1 ||
      fread(&epoch, sizeof(epoch), 1, f) != 1) {
    fprintf(stderr, "Invalid snapshot header in %s\n", path.c_str());
    fclose(f);
    return false;
  }
  if (num_programs == 0) {
    fprintf(stderr, "Snapshot %s has zero programs\n", path.c_str());
    fclose(f);
    return false;
  }

  soup_out->assign(num_programs * kSingleTapeSize, 0);
  if (fread(soup_out->data(), 1, soup_out->size(), f) != soup_out->size()) {
    fprintf(stderr, "Snapshot %s has truncated soup payload\n", path.c_str());
    fclose(f);
    return false;
  }
  fclose(f);
  *num_programs_out = num_programs;
  return true;
}

bool BuildStructuredPartnerPool(const std::vector<std::string> &snapshots,
                                std::vector<uint8_t> *pool_out) {
  pool_out->clear();
  if (snapshots.empty()) {
    return false;
  }

  for (const std::string &path : snapshots) {
    std::vector<uint8_t> soup;
    size_t num_programs = 0;
    if (!LoadSnapshotSoup(path, &soup, &num_programs)) {
      return false;
    }
    pool_out->insert(pool_out->end(), soup.begin(), soup.end());
    fprintf(stderr, "Loaded structured partner snapshot: %s (%zu programs)\n",
            path.c_str(), num_programs);
  }

  if (pool_out->empty() || pool_out->size() % kSingleTapeSize != 0) {
    fprintf(stderr, "Structured partner pool is invalid or empty\n");
    return false;
  }
  fprintf(stderr, "Structured partner pool size: %zu programs\n",
          pool_out->size() / kSingleTapeSize);
  return true;
}

constexpr size_t kMaxLoggedSelfreps = 5;

}  // namespace

namespace flags {
class BaseFlag {
 public:
  static std::unordered_map<std::string, BaseFlag *> *reg() {
    static std::unordered_map<std::string, BaseFlag *> r;
    return &r;
  }

  virtual size_t Parse(int argc, char **argv) = 0;
  virtual const char *Description() = 0;
  virtual void PrintValue() = 0;
  virtual bool ShouldPrintDescription(const std::string &flag) = 0;

  virtual ~BaseFlag() {}

  static void ParseCommandLine(int argc, char **argv) {
    auto help_and_exit = [&]() {
      fprintf(stderr, "%s flags\n\n", argv[0]);
      std::vector<std::string> all_flags;
      for (const auto &[f, _] : *reg()) {
        all_flags.push_back(f);
      }
      std::sort(all_flags.begin(), all_flags.end());
      for (const auto &f : all_flags) {
        if (!(*reg())[f]->ShouldPrintDescription(f)) {
          continue;
        }
        fprintf(stderr, "     %s: %s; default: ", f.c_str(),
                (*reg())[f]->Description());
        (*reg())[f]->PrintValue();
        fprintf(stderr, "\n");
      }
      std::exit(1);
    };

    for (int pos = 1; pos < argc; pos++) {
      if (argv[pos] == std::string("--help") ||
          argv[pos] == std::string("-h")) {
        help_and_exit();
      }

      auto iter = (*reg()).find(argv[pos]);
      if (iter == (*reg()).end()) {
        fprintf(stderr, "Unknown flag %s\n", argv[pos]);
        help_and_exit();
      }
      BaseFlag *parser = iter->second;
      pos += parser->Parse(argc - pos, argv + pos);
    }
  }
};

template <typename T>
struct Types;

template <>
struct Types<std::string> {
  static std::string Parse(const char *v) { return v; }
  static void Print(const std::string &v) { fprintf(stderr, "%s", v.c_str()); }
};

template <>
struct Types<double> {
  static double Parse(const char *v) { return std::stod(v); }
  static void Print(const double &v) { fprintf(stderr, "%8.5f", v); }
};

template <>
struct Types<size_t> {
  static size_t Parse(const char *v) { return std::stoul(v); }
  static void Print(const size_t &v) { fprintf(stderr, "%zu", v); }
};

template <typename T>
struct Types<std::optional<T>> {
  static std::optional<T> Parse(const char *v) { return Types<T>::Parse(v); }
  static void Print(const std::optional<T> &v) {
    if (v.has_value()) {
      Types<T>::Print(*v);
    } else {
      fprintf(stderr, "<no value>");
    }
  }
};

template <typename T>
class Flag : public BaseFlag {
 public:
  Flag(const char *opt, const char *noopt, T default_value,
       const char *description)
      : opt_(opt),
        noopt_(noopt),
        value_(std::move(default_value)),
        description_(description) {
    (*reg())[opt] = this;
    if constexpr (std::is_same_v<bool, T>) {
      (*reg())[noopt] = this;
    }
  }

  const T &Get() const { return value_; }

  size_t Parse(int argc, char **argv) override {
    if constexpr (std::is_same_v<bool, T>) {
      value_ = (argv[0] == std::string(opt_));
      return 0;
    } else {
      if (argc < 2) {
        fprintf(stderr, "missing argument for flag %s\n", argv[0]);
        std::exit(1);
      }
      value_ = Types<T>::Parse(argv[1]);
      return 1;
    }
  }
  const char *Description() override { return description_; }
  bool ShouldPrintDescription(const std::string &flag) override {
    return flag == opt_;
  }
  void PrintValue() override {
    if constexpr (std::is_same_v<bool, T>) {
      fprintf(stderr, "%s", value_ ? "true" : "false");
    } else {
      Types<T>::Print(value_);
    }
  }

 private:
  const char *opt_;
  const char *noopt_;
  T value_;
  const char *description_;
};

#define FLAG(type, name, default_value, description)                      \
  flags::Flag<type> FLAGS_##name("--" #name, "--no" #name, default_value, \
                                 description);

template <typename T>
const T &GetFlag(const Flag<T> &flag) {
  return flag.Get();
}

void ParseCommandLine(int argc, char **argv) {
  BaseFlag::ParseCommandLine(argc, argv);
}

}  // namespace flags

FLAG(std::optional<std::string>, run, std::nullopt, "run a program");
FLAG(size_t, run_steps, 32 * 1024, "max number of steps for running a program");
FLAG(bool, debug, false, "print execution step by step");
FLAG(size_t, num, 128 * 1024, "number of programs to evolve");
FLAG(std::optional<size_t>, max_epochs, std::nullopt, "max epochs");
FLAG(size_t, seed, 0, "seed");
FLAG(double, mutation_prob, 1.0 / (256 * 16), "mutation_prob");
FLAG(std::optional<double>, stopping_bpb, std::nullopt,
     "bits per byte below which to stop execution");
FLAG(std::optional<std::string>, initial_program, std::nullopt,
     "program to seed the soup with");
FLAG(std::optional<std::string>, log, std::nullopt, "log file");
FLAG(std::optional<std::string>, load, std::nullopt, "load a previous save");
FLAG(std::optional<std::string>, checkpoint_dir, std::nullopt,
     "directory to store checkpoints");
FLAG(std::optional<size_t>, reset_interval, std::nullopt, "reset interval");
FLAG(std::string, lang, "", "language to run");
FLAG(std::string, interaction_pattern, "",
     "file containing the allowed interactions, with two integers `a b` per "
     "line, representing that program `a` is allowed to interact with `b` (in "
     "that order).");
FLAG(bool, permute_programs, true,
     "do not shuffle programs between runs (cyclic interactions)");
FLAG(bool, fixed_shuffle, false, "deterministic shuffling pattern");
FLAG(bool, zero_init, false, "zero init");
FLAG(std::optional<std::string>, init_symbol_probs, std::nullopt,
     "comma-separated probabilities for BFF symbols in fixed order "
     "[]+-.,<>{}0; remaining mass is spread uniformly over the other 245 "
     "byte values and used during soup initialization / reinit");
FLAG(bool, random_partner_interaction, false,
     "interact with a fresh random program every epoch instead of another "
     "soup program");
FLAG(bool, structured_partner_interaction, false,
     "interact with a static partner pool sampled with replacement each epoch");
FLAG(std::string, structured_partner_snapshots, "",
     "comma-separated .dat snapshot paths used to build the structured partner "
     "pool");
FLAG(bool, focal_analysis, false,
     "log per-epoch focal Hamming-change statistics instead of standard CSV");
FLAG(std::optional<size_t>, focal_programs, std::nullopt,
     "number of programs (from index 0) included in focal analysis; default: "
     "num");
FLAG(std::optional<std::string>, focal_log, std::nullopt,
     "focal-analysis CSV log path");
FLAG(bool, eval_selfrep, false, "evaluate self replication in every epoch");
FLAG(bool, reinit_each_epoch, false,
     "reinitialize every program each epoch instead of letting them interact");
FLAG(bool, print_selfrep, false,
     "print any program whose self-replication score crosses the threshold");
FLAG(size_t, print_interval, 64, "interval between prints");
FLAG(std::optional<size_t>, log_interval, std::nullopt,
     "interval between log entries and evaluations");
FLAG(size_t, save_interval, 256, "interval between saves");
FLAG(size_t, clear_interval, 2048, "interval between clears");
FLAG(std::string, draw_to, "",
     "directory to save 1d-drawn frames to (must exist)");
FLAG(std::string, draw_to_2d, "",
     "directory to save 2d-drawn frames to (must exist, and num must "
     "be a square number)");
FLAG(size_t, grid_width_2d, 0, "width of the 2d grid");
FLAG(bool, disable_output, false, "disable printing to stdout");
FLAG(std::optional<size_t>, stopping_selfrep_count, std::nullopt,
     "stop when that many programs appear to be self-replicators");

int main(int argc, char **argv) {
  flags::ParseCommandLine(argc, argv);

  bool debug = GetFlag(FLAGS_debug);

  SimulationParams params;
  params.num_programs = GetFlag(FLAGS_num);
  params.reset_interval = GetFlag(FLAGS_reset_interval);
  params.seed = GetFlag(FLAGS_seed);
  params.load_from = GetFlag(FLAGS_load);
  params.mutation_prob = std::round(GetFlag(FLAGS_mutation_prob) * (1 << 30));
  params.permute_programs = GetFlag(FLAGS_permute_programs);
  params.fixed_shuffle = GetFlag(FLAGS_fixed_shuffle);
  params.zero_init = GetFlag(FLAGS_zero_init);
  auto init_symbol_probs = GetFlag(FLAGS_init_symbol_probs);
  params.eval_selfrep = GetFlag(FLAGS_eval_selfrep);
  params.reinit_each_epoch = GetFlag(FLAGS_reinit_each_epoch);
  params.print_selfrep = GetFlag(FLAGS_print_selfrep);
  params.random_partner_interaction =
      GetFlag(FLAGS_random_partner_interaction);
  params.structured_partner_interaction =
      GetFlag(FLAGS_structured_partner_interaction);
  params.focal_analysis = GetFlag(FLAGS_focal_analysis);
  params.focal_program_count =
      GetFlag(FLAGS_focal_programs).value_or(params.num_programs);
  params.save_to = GetFlag(FLAGS_checkpoint_dir);
  params.save_interval = GetFlag(FLAGS_save_interval);
  if (params.fixed_shuffle &&
      (params.num_programs & (params.num_programs - 1)) != 0) {
    fprintf(stderr, "#programs must be a power of two for fixed shuffle\n");
    return 1;
  }
  std::string interaction_pattern = GetFlag(FLAGS_interaction_pattern);
  if (params.fixed_shuffle && !interaction_pattern.empty()) {
    fprintf(stderr, "fixed shuffle and interaction pattern are incompatible\n");
    return 1;
  }
  if (params.random_partner_interaction && !interaction_pattern.empty()) {
    fprintf(stderr,
            "random partner interaction does not use interaction patterns\n");
    return 1;
  }
  if (params.structured_partner_interaction && !interaction_pattern.empty()) {
    fprintf(stderr,
            "structured partner interaction does not use interaction patterns\n");
    return 1;
  }
  if (params.random_partner_interaction && params.reinit_each_epoch) {
    fprintf(stderr,
            "random partner interaction cannot be combined with reinit_each_epoch\n");
    return 1;
  }
  if (params.structured_partner_interaction && params.reinit_each_epoch) {
    fprintf(
        stderr,
        "structured partner interaction cannot be combined with reinit_each_epoch\n");
    return 1;
  }
  if (params.random_partner_interaction &&
      params.structured_partner_interaction) {
    fprintf(stderr,
            "random_partner_interaction and structured_partner_interaction are "
            "mutually exclusive\n");
    return 1;
  }
  if (params.zero_init && init_symbol_probs.has_value()) {
    fprintf(stderr,
            "--zero_init and --init_symbol_probs are mutually exclusive\n");
    return 1;
  }
  if (init_symbol_probs.has_value()) {
    std::string error;
    if (!BuildInitByteCdf(*init_symbol_probs, &params.init_byte_cdf, &error)) {
      fprintf(stderr, "--init_symbol_probs invalid: %s\n", error.c_str());
      return 1;
    }
  }

  std::string structured_partner_snapshots_raw =
      GetFlag(FLAGS_structured_partner_snapshots);
  std::vector<std::string> structured_partner_snapshots =
      SplitCommaSeparatedPaths(structured_partner_snapshots_raw);
  if (params.structured_partner_interaction &&
      structured_partner_snapshots.empty()) {
    fprintf(stderr,
            "structured_partner_interaction requires "
            "--structured_partner_snapshots\n");
    return 1;
  }
  if (!params.structured_partner_interaction &&
      !structured_partner_snapshots.empty()) {
    fprintf(stderr,
            "--structured_partner_snapshots provided but "
            "--structured_partner_interaction is disabled\n");
    return 1;
  }
  if (params.structured_partner_interaction &&
      !BuildStructuredPartnerPool(structured_partner_snapshots,
                                  &params.structured_partner_pool)) {
    return 1;
  }

  constexpr size_t kColumnPadding1d = 4;
  size_t programs_per_column_1d = 0;
  size_t num_columns_1d = 0;
  std::string draw_to_1d = GetFlag(FLAGS_draw_to);
  if (!draw_to_1d.empty()) {
    programs_per_column_1d =
        params.num_programs /
        std::ceil(std::sqrt(params.num_programs /
                            (kSingleTapeSize + kColumnPadding1d)));
    num_columns_1d = (params.num_programs + programs_per_column_1d - 1) /
                     programs_per_column_1d;
  }
  std::vector<uint8_t> draw_buf_1d(
      programs_per_column_1d * 3 *
      (num_columns_1d * (kSingleTapeSize + kColumnPadding1d) -
       kColumnPadding1d));

  size_t grid_width_2d = 0;
  std::string draw_to_2d = GetFlag(FLAGS_draw_to_2d);
  if (!draw_to_2d.empty()) {
    if (GetFlag(FLAGS_grid_width_2d)) {
      grid_width_2d = GetFlag(FLAGS_grid_width_2d);
      if (params.num_programs % grid_width_2d != 0) {
        fprintf(stderr, "grid width must divide num_programs\n");
        return 1;
      }
    } else {
      grid_width_2d = std::sqrt(params.num_programs);
      if (grid_width_2d * grid_width_2d != params.num_programs) {
        fprintf(stderr, "number of programs must be a square\n");
        return 1;
      }
    }
  }
  static_assert(kSingleTapeSize == 64, "fix drawing if tapes are not 64 bytes");
  std::vector<uint8_t> draw_buf_2d(params.num_programs * 3 * kSingleTapeSize);
  std::vector<size_t> previous_selfrep_score(params.num_programs, 0);

  if (!interaction_pattern.empty()) {
    FILE *f = fopen(interaction_pattern.c_str(), "r");
    if (!f) {
      fprintf(stderr, "could not open interaction pattern file\n");
      return 1;
    }
    params.allowed_interactions.resize(params.num_programs);
    long long a, b;
    while (fscanf(f, "%lld%lld", &a, &b) == 2) {
      if (static_cast<size_t>(a) >= params.num_programs ||
          static_cast<size_t>(b) >= params.num_programs) {
        fprintf(stderr,
                "invalid interaction pattern: programs not in [0, n) range.\n");
        return 1;
      }
      params.allowed_interactions[a].push_back(b);
    }
    fclose(f);
  }

  std::optional<std::string> log_to = GetFlag(FLAGS_log);
  std::optional<std::string> focal_log_to = GetFlag(FLAGS_focal_log);
  size_t print_interval = GetFlag(FLAGS_print_interval);
  size_t log_interval = GetFlag(FLAGS_log_interval).value_or(print_interval);
  size_t clear_interval = GetFlag(FLAGS_clear_interval);
  std::optional<size_t> max_epochs = GetFlag(FLAGS_max_epochs);
  std::optional<size_t> stopping_bpb = GetFlag(FLAGS_stopping_bpb);
  std::optional<size_t> stopping_selfrep_count =
      GetFlag(FLAGS_stopping_selfrep_count);
  bool disable_output = GetFlag(FLAGS_disable_output);

  if (log_interval == 0) {
    fprintf(stderr, "log interval must be greater than zero\n");
    return 1;
  }
  if (print_interval == 0) {
    fprintf(stderr, "print interval must be greater than zero\n");
    return 1;
  }
  if (clear_interval == 0) {
    fprintf(stderr, "clear interval must be greater than zero\n");
    return 1;
  }

  if (params.focal_analysis) {
    if (params.eval_selfrep) {
      fprintf(stderr, "focal_analysis cannot be combined with eval_selfrep\n");
      return 1;
    }
    if (params.print_selfrep) {
      fprintf(stderr, "focal_analysis cannot be combined with print_selfrep\n");
      return 1;
    }
    if (stopping_selfrep_count.has_value()) {
      fprintf(stderr,
              "focal_analysis cannot be combined with stopping_selfrep_count\n");
      return 1;
    }
    if (log_to.has_value()) {
      fprintf(stderr,
              "focal_analysis does not support --log; use --focal_log only\n");
      return 1;
    }
    if (!focal_log_to.has_value()) {
      fprintf(stderr, "focal_analysis requires --focal_log\n");
      return 1;
    }
    if (params.focal_program_count == 0 ||
        params.focal_program_count > params.num_programs) {
      fprintf(stderr, "focal_programs must be in [1, num]\n");
      return 1;
    }
    if (log_interval != 1) {
      fprintf(stderr,
              "focal_analysis requires --log_interval 1 for per-epoch stats\n");
      return 1;
    }
  } else if (focal_log_to.has_value()) {
    fprintf(stderr, "--focal_log requires --focal_analysis\n");
    return 1;
  }

  if (stopping_selfrep_count.has_value() && !params.eval_selfrep) {
    fprintf(stderr, "stopping_selfrep_count requires eval_selfrep\n");
    return 1;
  }
  if (params.print_selfrep && !params.eval_selfrep) {
    fprintf(stderr, "print_selfrep requires eval_selfrep\n");
    return 1;
  }

  if (params.save_interval % log_interval != 0) {
    fprintf(stderr, "save interval must be divisible by log interval\n");
    return 1;
  }

  if (clear_interval % log_interval != 0) {
    fprintf(stderr, "clear interval must be divisible by log interval\n");
    return 1;
  }

  params.callback_interval = log_interval;

  auto run_flag = GetFlag(FLAGS_run);
  auto lang = GetFlag(FLAGS_lang);
  const LanguageInterface *language = GetLanguage(lang);
  if (run_flag.has_value()) {
    printf("%s", ResetColors());
    language->RunSingleProgram(run_flag.value(), GetFlag(FLAGS_run_steps),
                               debug);
  } else {
    FILE *logfile = nullptr;
    FILE *focal_logfile = nullptr;
    if (params.focal_analysis) {
      focal_logfile = CheckFopen(focal_log_to->c_str(), "w");
      fprintf(focal_logfile,
              "epoch,mean_hamming,median_hamming,p90_hamming,max_hamming\n");
    } else if (log_to.has_value()) {
      logfile = CheckFopen(log_to->c_str(), "w");
      if (params.eval_selfrep) {
        fprintf(logfile,
                "epoch,brotli_size,soup_size,higher_entropy,number_selfreps,"
                "selfrep_tapes,selfrep_scores,count_op_<,count_op_>,"
                "count_op_{,count_op_},count_op_+,count_op_-,count_op_.,"
                "\"count_op_,\",count_op_[,count_op_],count_zero,"
                "mean_step_count_per_interaction,frac_term_step_cap,"
                "frac_term_ip_out_of_bounds,frac_term_bracket_mismatch\n");
      } else {
        fprintf(logfile,
                "epoch,brotli_size,soup_size,higher_entropy,count_op_<,"
                "count_op_>,count_op_{,count_op_},count_op_+,count_op_-,"
                "count_op_.,\"count_op_,\",count_op_[,count_op_],count_zero,"
                "mean_step_count_per_interaction,frac_term_step_cap,"
                "frac_term_ip_out_of_bounds,frac_term_bracket_mismatch\n");
      }
    }

    auto callback = [&](const SimulationState &state) {
      size_t repl_count = 0;
      if (params.eval_selfrep) {
        for (size_t i = 0; i < state.replication_per_prog.size(); i++) {
          if (state.replication_per_prog[i] >= kSelfrepThreshold) {
            repl_count++;
          }
        }
      }
      int repl_count_display =
          params.eval_selfrep ? static_cast<int>(repl_count) : -1;
      // if (params.eval_selfrep && params.print_selfrep && !disable_output) {
      //   for (size_t i = 0; i < state.replication_per_prog.size(); ++i) {
      //     size_t current_score = state.replication_per_prog[i];
      //     bool crossed_threshold =
      //         previous_selfrep_score[i] < kSelfrepThreshold &&
      //         current_score >= kSelfrepThreshold;
      //     previous_selfrep_score[i] = current_score;
      //     if (!crossed_threshold) continue;
      //     printf(
      //         "%sDetected self-replicator candidate at epoch %zu program %zu "
      //         "score=%zu\n",
      //         ResetColors(), state.epoch, i, current_score);
      //     language->PrintProgram(kSingleTapeSize,
      //                            state.soup.data() + i * kSingleTapeSize,
      //                            kSingleTapeSize, nullptr, 0);
      //   }
      // }
      bool should_print = !disable_output && state.epoch % print_interval == 0;
      if (should_print) {
        if (state.epoch % clear_interval == 1) {
          printf("%s\033[2J\033[H", ResetColors());
        }
        printf(
            "%s\033[0;0H    Elapsed: %10.3f        ops: %23zu     "
            "MOps/s: %12.3f Epochs: %13zu ops/prog/epoch: %10.3f\n"
            "Brotli size: %10zu Brotli bpb: %23.4f bytes/prog: %12.4f     H0: "
            "%13.4f higher entropy: %10.6f number of repicators: %10d\n",
            ResetColors(), state.elapsed_s, state.total_ops, state.mops_s,
            state.epoch, state.ops_per_run, state.brotli_size, state.brotli_bpb,
            state.bytes_per_prog, state.h0, state.higher_entropy,
            repl_count_display);

        for (auto [s, f] : state.frequent_bytes) {
          printf("\033[37;1m%s%s %5.2f%% ", s.c_str(), ResetColors(),
                 f * 100.0);
        }
        printf("\n");
        for (auto [s, f] : state.uncommon_bytes) {
          printf("\033[37;1m%s%s %5.2f%% ", s.c_str(), ResetColors(),
                 f * 100.0);
        }
        printf("\n\n\n");

        for (size_t i = 0; i < std::min<size_t>(48, params.num_programs / 2);
             i++) {
          size_t separators[1] = {kSingleTapeSize};
          if (params.eval_selfrep) {
            printf("%02d %02d ", (int)state.replication_per_prog[2 * i],
                   (int)state.replication_per_prog[2 * i + 1]);
          }
          language->PrintProgram(2 * kSingleTapeSize,
                                 state.soup.data() + i * 2 * kSingleTapeSize,
                                 2 * kSingleTapeSize, separators, 1);
        }
        fflush(stdout);
      }

      if (focal_logfile) {
        fprintf(focal_logfile, "%zu,%.6f,%.6f,%.6f,%zu\n", state.epoch,
                state.mean_hamming, state.median_hamming, state.p90_hamming,
                state.max_hamming);
        fflush(focal_logfile);
      } else if (logfile) {
        if (params.eval_selfrep) {
          std::vector<std::pair<size_t, size_t>> best_selfreps;
          best_selfreps.reserve(kMaxLoggedSelfreps);
          for (size_t i = 0; i < state.replication_per_prog.size(); ++i) {
            size_t score = state.replication_per_prog[i];
            if (score < kSelfrepThreshold) {
              continue;
            }
            auto it = best_selfreps.begin();
            while (it != best_selfreps.end() &&
                   (it->second > score ||
                    (it->second == score && it->first <= i))) {
              ++it;
            }
            best_selfreps.insert(it, {i, score});
            if (best_selfreps.size() > kMaxLoggedSelfreps) {
              best_selfreps.pop_back();
            }
          }
          std::vector<std::string> best_tapes;
          std::vector<size_t> best_scores;
          best_tapes.reserve(best_selfreps.size());
          best_scores.reserve(best_selfreps.size());
          for (const auto &[idx, score] : best_selfreps) {
            best_scores.push_back(score);
            best_tapes.emplace_back(
                BytesToHex(state.soup.data() + idx * kSingleTapeSize,
                           kSingleTapeSize));
          }
          std::string tapes_field = JsonArray(best_tapes);
          std::string scores_field = JsonArray(best_scores);
          fprintf(logfile,
                  "%zu,%zu,%zu,%f,%zu,\"%s\",\"%s\",%zu,%zu,%zu,%zu,%zu,%zu,"
                  "%zu,%zu,%zu,%zu,%zu,%f,%f,%f,%f\n",
                  state.epoch, state.brotli_size,
                  state.soup.size() / kSingleTapeSize, state.higher_entropy,
                  repl_count, CsvEscape(tapes_field).c_str(),
                  CsvEscape(scores_field).c_str(), state.count_op_lt,
                  state.count_op_gt, state.count_op_lbrace,
                  state.count_op_rbrace, state.count_op_plus,
                  state.count_op_minus, state.count_op_dot,
                  state.count_op_comma, state.count_op_lbracket,
                  state.count_op_rbracket, state.count_zero,
                  state.mean_step_count_per_interaction,
                  state.frac_term_step_cap, state.frac_term_ip_out_of_bounds,
                  state.frac_term_bracket_mismatch);
        } else {
          fprintf(logfile,
                  "%zu,%zu,%zu,%f,%zu,%zu,%zu,%zu,%zu,%zu,%zu,%zu,%zu,%zu,%zu,"
                  "%f,%f,%f,%f\n",
                  state.epoch, state.brotli_size,
                  state.soup.size() / kSingleTapeSize, state.higher_entropy,
                  state.count_op_lt, state.count_op_gt, state.count_op_lbrace,
                  state.count_op_rbrace, state.count_op_plus,
                  state.count_op_minus, state.count_op_dot,
                  state.count_op_comma, state.count_op_lbracket,
                  state.count_op_rbracket, state.count_zero,
                  state.mean_step_count_per_interaction,
                  state.frac_term_step_cap, state.frac_term_ip_out_of_bounds,
                  state.frac_term_bracket_mismatch);
        }
        fflush(logfile);
      }

      auto write_ppm = [](const std::string &base, size_t frame, size_t xs,
                          size_t ys, const std::vector<uint8_t> &data) {
        assert(data.size() == xs * ys * 3);
        std::string out_path(base.size() + 64, 0);
        out_path.resize(snprintf(out_path.data(), out_path.size(),
                                 "%s/%012lld.ppm", base.c_str(),
                                 (long long)frame));
        FILE *f = CheckFopen(out_path.c_str(), "w");
        fprintf(f, "P6\n%lld %lld\n255\n", (long long)xs, (long long)ys);
        fwrite(data.data(), 1, data.size(), f);
        fclose(f);
      };

      if (!draw_to_1d.empty()) {
        size_t xs = (kColumnPadding1d + kSingleTapeSize) * num_columns_1d -
                    kColumnPadding1d;
        for (size_t i = 0; i < params.num_programs; i++) {
          size_t column_id = i / programs_per_column_1d;
          size_t column_off = i % programs_per_column_1d;
          size_t x = column_id * (kColumnPadding1d + kSingleTapeSize);
          size_t y = column_off;
          for (size_t j = 0; j < kSingleTapeSize; j++) {
            memcpy(
                &draw_buf_1d[(y * xs + x + j) * 3],
                state.byte_colors[state.soup[i * kSingleTapeSize + j]].data(),
                3);
          }
        }
        write_ppm(draw_to_1d, state.epoch, xs, programs_per_column_1d,
                  draw_buf_1d);
      }

      if (!draw_to_2d.empty()) {
        static_assert(kSingleTapeSize == 64,
                      "fix drawing if tapes are not 64 bytes");
        size_t xs = grid_width_2d * 8;
        for (size_t i = 0; i < params.num_programs; i++) {
          size_t x = i % grid_width_2d;
          size_t y = i / grid_width_2d;
          for (size_t j = 0; j < kSingleTapeSize; j++) {
            size_t ix = j % 8;
            size_t iy = j / 8;
            size_t p = ((y * 8 + iy) * xs + x * 8 + ix) * 3;
            memcpy(
                &draw_buf_2d[p],
                state.byte_colors[state.soup[i * kSingleTapeSize + j]].data(),
                3);
            if (ix == 0 || iy == 0) {
              for (size_t c = 0; c < 3; ++c) {
                draw_buf_2d[p + c] = std::max(draw_buf_2d[p + c] - 32, 0);
              }
            }
          }
        }
        write_ppm(draw_to_2d, state.epoch, xs,
                  params.num_programs / grid_width_2d * 8, draw_buf_2d);
      }

      if (max_epochs.has_value() && state.epoch > *max_epochs) {
        return true;
      }
      if (stopping_bpb.has_value() && state.brotli_bpb < *stopping_bpb) {
        return true;
      };
      if (stopping_selfrep_count.has_value() &&
          repl_count >= *stopping_selfrep_count) {
        return true;
      }
      return false;
    };

    std::optional<std::string> initial_program = GetFlag(FLAGS_initial_program);
    language->RunSimulation(params, initial_program, callback);

    if (logfile) {
      fclose(logfile);
    }
    if (focal_logfile) {
      fclose(focal_logfile);
    }
  }
}
