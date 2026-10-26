// Console app running inference of the cs336_basics TransformerLM on the CPU.
//
//   cs336_infer --prompt "Once upon a time" --max_steps 128 --temperature 0.8 --top_p 0.95

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif
#ifdef _WIN32
#include <windows.h>
#endif

#include "model_config.h"
#include "tokenizer.h"
#include "transformer_lm.h"
#include "weights_file.h"

namespace {

struct Options {
  std::string model_path = "models/model.bin";
  std::string tokenizer_path = "models/tokenizer.bin";
  std::string prompt = "Once";
  float temperature = 0.0f;
  float top_p = 0.95f;
  int max_steps = 256;
  unsigned seed = 0;
  bool seed_given = false;
  int threads = 0;
  bool print_tokens = false;
  bool show_ids = false;
  std::string dump_logits_path;
};

void print_usage(const char* program) {
  std::fprintf(stderr,
               "Usage: %s [options]\n"
               "  --prompt TEXT         UTF-8 prompt (default \"Once\")\n"
               "  --max_steps N         maximum number of generated tokens (default 256)\n"
               "  --temperature T       sampling temperature, 0 = greedy (default 0)\n"
               "  --top_p P             nucleus sampling threshold (default 0.95)\n"
               "  --seed N              random seed (default: time based)\n"
               "  --threads N           OpenMP threads (ignored by builds without OpenMP)\n"
               "  --model PATH          model.bin (default models/model.bin)\n"
               "  --tokenizer PATH      tokenizer.bin (default models/tokenizer.bin)\n"
               "  --print-tokens        only print the token ids of the prompt\n"
               "  --show-ids            also print the prompt + generated token ids on stderr\n"
               "  --dump-logits FILE    write the float32 logits following the prompt to FILE and exit\n",
               program);
}

bool parse_args(int argc, char** argv, Options& o) {
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    const auto value = [&]() -> const char* {
      if (i + 1 >= argc) {
        std::fprintf(stderr, "missing value for %s\n", arg.c_str());
        std::exit(2);
      }
      return argv[++i];
    };
    if (arg == "--prompt") o.prompt = value();
    else if (arg == "--max_steps") o.max_steps = std::atoi(value());
    else if (arg == "--temperature") o.temperature = static_cast<float>(std::atof(value()));
    else if (arg == "--top_p") o.top_p = static_cast<float>(std::atof(value()));
    else if (arg == "--seed") { o.seed = static_cast<unsigned>(std::strtoul(value(), nullptr, 10)); o.seed_given = true; }
    else if (arg == "--threads") o.threads = std::atoi(value());
    else if (arg == "--model") o.model_path = value();
    else if (arg == "--tokenizer") o.tokenizer_path = value();
    else if (arg == "--print-tokens") o.print_tokens = true;
    else if (arg == "--show-ids") o.show_ids = true;
    else if (arg == "--dump-logits") o.dump_logits_path = value();
    else if (arg == "--help" || arg == "-h") return false;
    else {
      std::fprintf(stderr, "unknown option %s\n", arg.c_str());
      return false;
    }
  }
  if (o.temperature < 0.0f || o.top_p <= 0.0f || o.top_p > 1.0f || o.max_steps < 0) {
    std::fprintf(stderr, "invalid sampling parameters (temperature >= 0, 0 < top_p <= 1, max_steps >= 0)\n");
    return false;
  }
  return true;
}

void print_ids(const char* label, const std::vector<int>& ids, std::FILE* out) {
  std::fprintf(out, "%s", label);
  for (std::size_t i = 0; i < ids.size(); ++i) std::fprintf(out, "%s%d", i ? " " : "", ids[i]);
  std::fprintf(out, "\n");
}

int run(const Options& options) {
  const cs336::Tokenizer tokenizer(options.tokenizer_path);
  const std::vector<int> prompt = tokenizer.encode(options.prompt);

  if (options.print_tokens) {
    print_ids("", prompt, stdout);
    return 0;
  }
  if (prompt.empty()) {
    std::fprintf(stderr, "error: the prompt is empty\n");
    return 1;
  }
  if (static_cast<int>(prompt.size()) > cs336::MAX_SEQ_LEN) {
    std::fprintf(stderr, "error: the prompt has %zu tokens but max_seq_len is %d\n", prompt.size(), cs336::MAX_SEQ_LEN);
    return 1;
  }

#ifdef _OPENMP
  if (options.threads > 0) omp_set_num_threads(options.threads);
  std::fprintf(stderr, "[cs336_infer] OpenMP threads: %d\n", omp_get_max_threads());
#else
  if (options.threads > 0) std::fprintf(stderr, "[cs336_infer] warning: built without OpenMP, --threads is ignored\n");
  std::fprintf(stderr, "[cs336_infer] single-threaded build\n");
#endif

  const auto load_start = std::chrono::steady_clock::now();
  cs336::TransformerLM model;
  {
    cs336::WeightsFile weights(options.model_path);
    model.load_weights(weights);
  }
  const double load_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - load_start).count();
  std::fprintf(stderr, "[cs336_infer] model loaded in %.2f s\n", load_seconds);

  if (!options.dump_logits_path.empty()) {
    const float* logits = model.forward(prompt.data(), static_cast<int>(prompt.size()));
    std::FILE* f = std::fopen(options.dump_logits_path.c_str(), "wb");
    if (!f || std::fwrite(logits, sizeof(float), cs336::VOCAB_SIZE, f) != static_cast<std::size_t>(cs336::VOCAB_SIZE)) {
      std::fprintf(stderr, "error: cannot write %s\n", options.dump_logits_path.c_str());
      return 1;
    }
    std::fclose(f);
    return 0;
  }

  const unsigned seed = options.seed_given
                            ? options.seed
                            : static_cast<unsigned>(std::chrono::system_clock::now().time_since_epoch().count());
  cs336::Sampler sampler(options.temperature, options.top_p, seed);

  std::fputs(options.prompt.c_str(), stdout);
  std::fflush(stdout);
  cs336::Utf8StreamDecoder decoder;
  const int eos = tokenizer.eos_id();
  cs336::GenerateStats stats;
  const std::vector<int> sequence = model.generate(
      prompt, eos, sampler, options.max_steps,
      [&](int token) {
        if (token == eos) return;
        std::fputs(decoder.push(tokenizer.token_bytes(token)).c_str(), stdout);
        std::fflush(stdout);
      },
      &stats);
  std::fputs(decoder.flush().c_str(), stdout);
  std::fputc('\n', stdout);

  if (options.show_ids) print_ids("ids: ", sequence, stderr);
  std::fprintf(stderr, "[cs336_infer] prefill: %d tokens in %.3f s (%.1f tok/s)\n", stats.prompt_tokens,
               stats.prefill_seconds, stats.prefill_seconds > 0 ? stats.prompt_tokens / stats.prefill_seconds : 0.0);
  std::fprintf(stderr, "[cs336_infer] decode: %d tokens in %.3f s (%.1f tok/s)\n", stats.generated_tokens,
               stats.decode_seconds, stats.decode_seconds > 0 ? stats.generated_tokens / stats.decode_seconds : 0.0);
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
#ifdef _WIN32
  SetConsoleOutputCP(CP_UTF8);  // the app prints UTF-8 bytes
#endif
  Options options;
  if (!parse_args(argc, argv, options)) {
    print_usage(argv[0]);
    return 2;
  }
  try {
    return run(options);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }
}
