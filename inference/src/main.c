// Console app running inference of the cs336_basics TransformerLM on the CPU.
//
//   cs336_infer --prompt "Once upon a time" --max_steps 128 --temperature 0.8 --top_p 0.95

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "model_config.h"
#include "platform.h"
#include "sampler.h"
#include "tokenizer.h"
#include "transformer_lm.h"
#include "weights_file.h"

typedef struct {
  const char *model_path;
  const char *tokenizer_path;
  const char *prompt;
  const char *prompt_file;
  const char *output_file;
  float temperature;
  float top_p;
  int max_steps;
  unsigned seed;
  int seed_given;
  int threads;
  int print_tokens;
  int show_ids;
  const char *dump_logits_path;
} Options;

static void print_usage(const char *program) {
  fprintf(stderr,
          "Usage: %s [options]\n"
          "  --prompt TEXT         UTF-8 prompt (default \"Once\")\n"
          "  --prompt-file FILE    read the UTF-8 prompt from FILE instead (exact bytes, a UTF-8 BOM is skipped)\n"
          "  --output-file FILE    write the generated text (raw UTF-8 bytes) to FILE instead of stdout\n"
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

// Returns the value following the option at argv[*i], or NULL (after a message) if there is none.
static const char *option_value(int argc, char **argv, int *i) {
  if (*i + 1 >= argc) {
    fprintf(stderr, "missing value for %s\n", argv[*i]);
    return NULL;
  }
  return argv[++*i];
}

static int parse_args(int argc, char **argv, Options *o) {
  o->model_path = "models/model.bin";
  o->tokenizer_path = "models/tokenizer.bin";
  o->prompt = "Once";
  o->temperature = 0.0f;
  o->top_p = 0.95f;
  o->max_steps = 256;
  for (int i = 1; i < argc; ++i) {
    const char *arg = argv[i];
    const char *v;
    if (strcmp(arg, "--print-tokens") == 0) {
      o->print_tokens = 1;
    } else if (strcmp(arg, "--show-ids") == 0) {
      o->show_ids = 1;
    } else if (strcmp(arg, "--help") == 0 || strcmp(arg, "-h") == 0) {
      return -1;
    } else if (strcmp(arg, "--prompt") == 0 || strcmp(arg, "--prompt-file") == 0 || strcmp(arg, "--output-file") == 0 ||
               strcmp(arg, "--max_steps") == 0 || strcmp(arg, "--temperature") == 0 || strcmp(arg, "--top_p") == 0 ||
               strcmp(arg, "--seed") == 0 || strcmp(arg, "--threads") == 0 || strcmp(arg, "--model") == 0 ||
               strcmp(arg, "--tokenizer") == 0 || strcmp(arg, "--dump-logits") == 0) {
      v = option_value(argc, argv, &i);
      if (!v) return -2;
      if (strcmp(arg, "--prompt") == 0) o->prompt = v;
      else if (strcmp(arg, "--prompt-file") == 0) o->prompt_file = v;
      else if (strcmp(arg, "--output-file") == 0) o->output_file = v;
      else if (strcmp(arg, "--max_steps") == 0) o->max_steps = atoi(v);
      else if (strcmp(arg, "--temperature") == 0) o->temperature = (float)atof(v);
      else if (strcmp(arg, "--top_p") == 0) o->top_p = (float)atof(v);
      else if (strcmp(arg, "--seed") == 0) { o->seed = (unsigned)strtoul(v, NULL, 10); o->seed_given = 1; }
      else if (strcmp(arg, "--threads") == 0) o->threads = atoi(v);
      else if (strcmp(arg, "--model") == 0) o->model_path = v;
      else if (strcmp(arg, "--tokenizer") == 0) o->tokenizer_path = v;
      else o->dump_logits_path = v;
    } else {
      fprintf(stderr, "unknown option %s\n", arg);
      return -1;
    }
  }
  if (o->temperature < 0.0f || o->top_p <= 0.0f || o->top_p > 1.0f || o->max_steps < 0) {
    fprintf(stderr, "invalid sampling parameters (temperature >= 0, 0 < top_p <= 1, max_steps >= 0)\n");
    return -1;
  }
  return 0;
}

static void print_ids(const char *label, const int *ids, size_t n, FILE *out) {
  fputs(label, out);
  for (size_t i = 0; i < n; ++i) fprintf(out, "%s%d", i ? " " : "", ids[i]);
  fputc('\n', out);
}

// Reads a whole file; a leading UTF-8 byte order mark is dropped.
static int read_file(const char *path, CsBuf *out) {
  FILE *f = fopen(path, "rb");
  if (!f) {
    cs_set_error("cannot open file: %s", path);
    return -1;
  }
  unsigned char chunk[4096];
  size_t got;
  while ((got = fread(chunk, 1, sizeof chunk, f)) > 0) {
    if (csbuf_append(out, chunk, got) != 0) {
      fclose(f);
      return -1;
    }
  }
  const int failed = ferror(f);
  fclose(f);
  if (failed) {
    cs_set_error("cannot read file: %s", path);
    return -1;
  }
  if (out->len >= 3 && memcmp(out->data, "\xEF\xBB\xBF", 3) == 0) {
    memmove(out->data, out->data + 3, out->len - 3);
    out->len -= 3;
  }
  return 0;
}

typedef struct {
  const Tokenizer *tokenizer;
  Utf8StreamDecoder decoder;
  CsBuf text;
  FILE *out;
  int eos;
  int failed;
} StreamState;

static void write_text(StreamState *s) {
  if (s->text.len) fwrite(s->text.data, 1, s->text.len, s->out);
  fflush(s->out);
  csbuf_clear(&s->text);
}

static void on_token(int token, void *user) {
  StreamState *s = (StreamState *)user;
  if (token == s->eos || s->failed) return;
  size_t length = 0;
  const unsigned char *bytes = tokenizer_token_bytes(s->tokenizer, token, &length);
  if (!bytes || utf8_stream_push(&s->decoder, bytes, length, &s->text) != 0) {
    s->failed = 1;
    return;
  }
  write_text(s);
}

static int run(const Options *options) {
  int status = -1;
  Tokenizer tokenizer;
  int tokenizer_ready = 0;
  TransformerLM model;
  int model_ready = 0;
  Sampler sampler;
  int sampler_ready = 0;
  CsBuf prompt_text = {0}, prompt_ids = {0}, sequence = {0};
  StreamState stream;
  memset(&stream, 0, sizeof stream);
  FILE *out = stdout;

  const char *prompt = options->prompt;
  size_t prompt_size = strlen(prompt);
  if (options->prompt_file) {
    if (read_file(options->prompt_file, &prompt_text) != 0) goto done;
    prompt = (const char *)prompt_text.data;
    prompt_size = prompt_text.len;
  }

  if (tokenizer_open(&tokenizer, options->tokenizer_path) != 0) goto done;
  tokenizer_ready = 1;
  if (tokenizer_encode(&tokenizer, prompt, prompt_size, &prompt_ids) != 0) goto done;
  const int *ids = (const int *)prompt_ids.data;
  const int num_prompt = (int)(prompt_ids.len / sizeof(int));

  if (options->print_tokens) {
    print_ids("", ids, (size_t)num_prompt, stdout);
    status = 0;
    goto done;
  }
  if (num_prompt == 0) {
    cs_set_error("the prompt is empty");
    goto done;
  }
  if (num_prompt > MAX_SEQ_LEN) {
    cs_set_error("the prompt has %d tokens but max_seq_len is %d", num_prompt, MAX_SEQ_LEN);
    goto done;
  }

#ifdef _OPENMP
  if (options->threads > 0) omp_set_num_threads(options->threads);
  fprintf(stderr, "[cs336_infer] OpenMP threads: %d\n", omp_get_max_threads());
#else
  if (options->threads > 0) fprintf(stderr, "[cs336_infer] warning: built without OpenMP, --threads is ignored\n");
  fprintf(stderr, "[cs336_infer] single-threaded build\n");
#endif

  const double load_start = cs_now_seconds();
  model_ready = 1;
  if (transformer_lm_init(&model) != 0) goto done;
  {
    WeightsFile weights;
    if (weights_file_open(&weights, options->model_path) != 0) goto done;
    const int loaded = transformer_lm_load_weights(&model, &weights);
    weights_file_close(&weights);
    if (loaded != 0) goto done;
  }
  fprintf(stderr, "[cs336_infer] model loaded in %.2f s\n", cs_now_seconds() - load_start);

  if (options->dump_logits_path) {
    const float *logits = NULL;
    if (transformer_lm_forward(&model, ids, num_prompt, &logits) != 0) goto done;
    FILE *f = fopen(options->dump_logits_path, "wb");
    if (!f || fwrite(logits, sizeof(float), VOCAB_SIZE, f) != (size_t)VOCAB_SIZE) {
      if (f) fclose(f);
      cs_set_error("cannot write %s", options->dump_logits_path);
      goto done;
    }
    fclose(f);
    status = 0;
    goto done;
  }

  const unsigned seed = options->seed_given ? options->seed : cs_time_seed();
  if (sampler_init(&sampler, options->temperature, options->top_p, seed, VOCAB_SIZE) != 0) goto done;
  sampler_ready = 1;

  if (options->output_file) {
    out = fopen(options->output_file, "wb");
    if (!out) {
      cs_set_error("cannot write %s", options->output_file);
      out = stdout;
      goto done;
    }
  }
  fwrite(prompt, 1, prompt_size, out);
  fflush(out);

  stream.tokenizer = &tokenizer;
  stream.out = out;
  stream.eos = tokenizer_eos_id(&tokenizer);
  utf8_stream_init(&stream.decoder);
  GenerateStats stats;
  if (transformer_lm_generate(&model, ids, num_prompt, stream.eos, &sampler, options->max_steps, on_token, &stream, &stats,
                              &sequence) != 0) {
    goto done;
  }
  if (stream.failed) goto done;
  if (utf8_stream_flush(&stream.decoder, &stream.text) != 0) goto done;
  fwrite(stream.text.data, 1, stream.text.len, out);
  fputc('\n', out);
  fflush(out);

  if (options->show_ids) print_ids("ids: ", (const int *)sequence.data, sequence.len / sizeof(int), stderr);
  fprintf(stderr, "[cs336_infer] prefill: %d tokens in %.3f s (%.1f tok/s)\n", stats.prompt_tokens, stats.prefill_seconds,
          stats.prefill_seconds > 0 ? stats.prompt_tokens / stats.prefill_seconds : 0.0);
  fprintf(stderr, "[cs336_infer] decode: %d tokens in %.3f s (%.1f tok/s)\n", stats.generated_tokens, stats.decode_seconds,
          stats.decode_seconds > 0 ? stats.generated_tokens / stats.decode_seconds : 0.0);
  status = 0;

done:
  if (out != stdout) fclose(out);
  if (sampler_ready) sampler_free(&sampler);
  if (model_ready) transformer_lm_free(&model);
  if (tokenizer_ready) tokenizer_free(&tokenizer);
  csbuf_free(&stream.text);
  csbuf_free(&sequence);
  csbuf_free(&prompt_ids);
  csbuf_free(&prompt_text);
  if (status != 0) fprintf(stderr, "error: %s\n", cs_get_error());
  return status == 0 ? 0 : 1;
}

int main(int argc, char **argv) {
  // First thing: on a 32-bit build any SSE instruction would crash on a CPU without it.
  if (!cs_cpu_supported()) {
    fputs("error: this CPU does not support SSE (Pentium III or later is required)\n", stderr);
    return 1;
  }
  Options options;
  memset(&options, 0, sizeof options);
  if (parse_args(argc, argv, &options) != 0) {
    print_usage(argv[0]);
    return 2;
  }
  return run(&options);
}
