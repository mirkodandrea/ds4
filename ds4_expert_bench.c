/*
 * ds4_expert_bench.c — Pure-compute expert throughput benchmark.
 *
 * Loads the model in expert-only mode and repeatedly calls
 * ds4_engine_compute_experts[_batch] with synthetic activations,
 * measuring raw compute throughput without any network overhead.
 *
 * Usage:
 *   ds4-expert-bench <model.gguf> [options]
 *
 * Options:
 *   --threads N       CPU threads (default: auto)
 *   --experts S-E     Expert range (default: 0-256)
 *   --layer L         Layer to benchmark (default: 5)
 *   --n-experts N     Experts per token (default: 6)
 *   --batch N         Batch size in tokens (default: 1)
 *   --iters N         Number of iterations (default: 100)
 *   --warmup N        Warmup iterations (default: 10)
 */

#include "ds4.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define DS4_BENCH_N_EMBD       4096
#define DS4_BENCH_N_EXPERT     256
#define DS4_BENCH_N_EXPERT_USED 6
#define DS4_BENCH_QK_K         256
#define DS4_BENCH_Q8K_BLOCK_SIZE 292  /* sizeof(block_q8_K): 4 + 256 + 32 = 292 */
#define DS4_BENCH_Q8K_BLOCKS   (DS4_BENCH_N_EMBD / DS4_BENCH_QK_K)
#define DS4_BENCH_Q8K_BYTES    (DS4_BENCH_Q8K_BLOCKS * DS4_BENCH_Q8K_BLOCK_SIZE)

static double bench_now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

/* Fill Q8_K activation with pseudo-random data that won't cause NaN. */
static void fill_random_q8k(uint8_t *buf, size_t n_tokens) {
    uint32_t rng = 0xDEADBEEF;
    for (size_t i = 0; i < n_tokens * DS4_BENCH_Q8K_BYTES; i++) {
        rng = rng * 1103515245u + 12345u;
        buf[i] = (uint8_t)(rng >> 16);
    }
    /* Set valid f32 scale values in each Q8_K block header. */
    for (size_t t = 0; t < n_tokens; t++) {
        for (int b = 0; b < DS4_BENCH_Q8K_BLOCKS; b++) {
            float scale = 0.01f;
            size_t off = t * DS4_BENCH_Q8K_BYTES + (size_t)b * DS4_BENCH_Q8K_BLOCK_SIZE;
            memcpy(buf + off, &scale, sizeof(float));
        }
    }
}

static void usage(void) {
    fprintf(stderr,
        "Usage: ds4-expert-bench <model.gguf> [options]\n"
        "\n"
        "Pure-compute expert throughput benchmark. No network overhead.\n"
        "\n"
        "Options:\n"
        "  --threads N       CPU threads (default: auto)\n"
        "  --experts S-E     Expert range (default: 0-256)\n"
        "  --layer L         Layer to benchmark (default: 5)\n"
        "  --n-experts N     Experts per token, 1-%d (default: %d)\n"
        "  --batch N         Batch size in tokens (default: 1)\n"
        "  --iters N         Timed iterations (default: 100)\n"
        "  --warmup N        Warmup iterations (default: 10)\n"
        "\n", DS4_BENCH_N_EXPERT_USED, DS4_BENCH_N_EXPERT_USED);
    exit(1);
}

int main(int argc, char **argv) {
    const char *model_path = NULL;
    int n_threads = 0;
    uint16_t expert_start = 0;
    uint16_t expert_end = DS4_BENCH_N_EXPERT;
    int layer = 5;
    int n_experts = DS4_BENCH_N_EXPERT_USED;
    int batch_size = 1;
    int n_iters = 100;
    int n_warmup = 10;

    for (int i = 1; i < argc; i++) {
        if (argv[i][0] != '-' && !model_path) {
            model_path = argv[i];
        } else if (strcmp(argv[i], "--threads") == 0 && i + 1 < argc) {
            n_threads = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--experts") == 0 && i + 1 < argc) {
            char *dash = strchr(argv[++i], '-');
            if (!dash) usage();
            *dash = '\0';
            expert_start = (uint16_t)atoi(argv[i]);
            expert_end = (uint16_t)atoi(dash + 1);
        } else if (strcmp(argv[i], "--layer") == 0 && i + 1 < argc) {
            layer = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--n-experts") == 0 && i + 1 < argc) {
            n_experts = atoi(argv[++i]);
            if (n_experts < 1 || n_experts > DS4_BENCH_N_EXPERT_USED) {
                fprintf(stderr, "n-experts must be 1-%d\n", DS4_BENCH_N_EXPERT_USED);
                return 1;
            }
        } else if (strcmp(argv[i], "--batch") == 0 && i + 1 < argc) {
            batch_size = atoi(argv[++i]);
            if (batch_size < 1) { fprintf(stderr, "batch must be >= 1\n"); return 1; }
        } else if (strcmp(argv[i], "--iters") == 0 && i + 1 < argc) {
            n_iters = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--warmup") == 0 && i + 1 < argc) {
            n_warmup = atoi(argv[++i]);
        } else {
            usage();
        }
    }

    if (!model_path) usage();

    fprintf(stderr, "ds4-expert-bench: loading %s (expert-only mode)\n", model_path);

    ds4_engine_options opts = {
        .model_path = model_path,
        .backend = DS4_BACKEND_CPU,
        .n_threads = n_threads,
        .expert_only = true,
    };
    ds4_engine *engine = NULL;
    if (ds4_engine_open(&engine, &opts) != 0) {
        fprintf(stderr, "ds4-expert-bench: failed to load model\n");
        return 1;
    }

    /* Pick expert IDs within owned range. */
    uint16_t expert_ids[DS4_BENCH_N_EXPERT_USED];
    float expert_weights[DS4_BENCH_N_EXPERT_USED];
    for (int i = 0; i < n_experts; i++) {
        expert_ids[i] = expert_start + (uint16_t)(i % (expert_end - expert_start));
        expert_weights[i] = 1.0f / (float)n_experts;
    }

    fprintf(stderr, "\n");
    fprintf(stderr, "Configuration:\n");
    fprintf(stderr, "  Layer:           %d\n", layer);
    fprintf(stderr, "  Experts/token:   %d\n", n_experts);
    fprintf(stderr, "  Batch size:      %d\n", batch_size);
    fprintf(stderr, "  Expert range:    %u-%u\n", expert_start, expert_end);
    fprintf(stderr, "  Warmup iters:    %d\n", n_warmup);
    fprintf(stderr, "  Timed iters:     %d\n", n_iters);
    fprintf(stderr, "\n");

    if (batch_size == 1) {
        /* ---- Single-token benchmark ---- */
        uint8_t *xq = malloc(DS4_BENCH_Q8K_BYTES);
        float *out = malloc((size_t)DS4_BENCH_N_EMBD * sizeof(float));
        if (!xq || !out) { fprintf(stderr, "OOM\n"); return 1; }
        fill_random_q8k(xq, 1);

        /* Warmup */
        fprintf(stderr, "Warming up (%d iters)...\n", n_warmup);
        for (int i = 0; i < n_warmup; i++) {
            ds4_engine_compute_experts(engine, (uint8_t)layer, xq,
                                       expert_ids, expert_weights, n_experts, out);
        }

        /* Timed run */
        fprintf(stderr, "Benchmarking (%d iters)...\n", n_iters);
        double *times = malloc((size_t)n_iters * sizeof(double));

        double total = 0.0;
        for (int i = 0; i < n_iters; i++) {
            double t0 = bench_now();
            ds4_engine_compute_experts(engine, (uint8_t)layer, xq,
                                       expert_ids, expert_weights, n_experts, out);
            double t1 = bench_now();
            times[i] = (t1 - t0) * 1000.0;  /* ms */
            total += times[i];
        }

        /* Stats */
        double mean = total / n_iters;
        double min_t = times[0], max_t = times[0];
        double var = 0.0;
        for (int i = 0; i < n_iters; i++) {
            if (times[i] < min_t) min_t = times[i];
            if (times[i] > max_t) max_t = times[i];
            var += (times[i] - mean) * (times[i] - mean);
        }
        double stddev = sqrt(var / n_iters);

        /* Sort for percentiles */
        for (int i = 0; i < n_iters - 1; i++)
            for (int j = i + 1; j < n_iters; j++)
                if (times[j] < times[i]) { double tmp = times[i]; times[i] = times[j]; times[j] = tmp; }
        double p50 = times[n_iters / 2];
        double p95 = times[(int)(n_iters * 0.95)];
        double p99 = times[(int)(n_iters * 0.99)];

        double tok_per_sec = 1000.0 / mean;

        fprintf(stderr, "\n");
        fprintf(stderr, "=== Single-token expert compute (layer %d, %d experts) ===\n", layer, n_experts);
        fprintf(stderr, "  Mean:       %8.3f ms  (%.1f tok/s)\n", mean, tok_per_sec);
        fprintf(stderr, "  Median:     %8.3f ms\n", p50);
        fprintf(stderr, "  Min:        %8.3f ms\n", min_t);
        fprintf(stderr, "  Max:        %8.3f ms\n", max_t);
        fprintf(stderr, "  Stddev:     %8.3f ms\n", stddev);
        fprintf(stderr, "  P95:        %8.3f ms\n", p95);
        fprintf(stderr, "  P99:        %8.3f ms\n", p99);
        fprintf(stderr, "\n");

        /* Throughput projections */
        fprintf(stderr, "  Throughput projection (43 layers):\n");
        fprintf(stderr, "    Expert-only latency/token: %8.3f ms  (%.1f tok/s)\n",
                mean * 43, 1000.0 / (mean * 43));
        fprintf(stderr, "\n");

        free(times);
        free(xq);
        free(out);
    } else {
        /* ---- Batched benchmark ---- */
        uint8_t *xq_all = malloc((size_t)batch_size * DS4_BENCH_Q8K_BYTES);
        float *out_all = malloc((size_t)batch_size * DS4_BENCH_N_EMBD * sizeof(float));
        uint16_t *all_ids = malloc((size_t)batch_size * n_experts * sizeof(uint16_t));
        float *all_wts = malloc((size_t)batch_size * n_experts * sizeof(float));
        uint8_t *token_n = malloc((size_t)batch_size * sizeof(uint8_t));
        if (!xq_all || !out_all || !all_ids || !all_wts || !token_n) {
            fprintf(stderr, "OOM\n"); return 1;
        }
        fill_random_q8k(xq_all, (size_t)batch_size);

        for (int t = 0; t < batch_size; t++) {
            token_n[t] = (uint8_t)n_experts;
            for (int e = 0; e < n_experts; e++) {
                all_ids[t * n_experts + e] = expert_ids[e];
                all_wts[t * n_experts + e] = expert_weights[e];
            }
        }

        /* Warmup */
        fprintf(stderr, "Warming up (%d iters, batch=%d)...\n", n_warmup, batch_size);
        for (int i = 0; i < n_warmup; i++) {
            ds4_engine_compute_experts_batch(engine, (uint8_t)layer,
                xq_all, all_ids, all_wts, token_n,
                batch_size, n_experts, out_all);
        }

        /* Timed run */
        fprintf(stderr, "Benchmarking (%d iters, batch=%d)...\n", n_iters, batch_size);
        double *times = malloc((size_t)n_iters * sizeof(double));

        double total = 0.0;
        for (int i = 0; i < n_iters; i++) {
            double t0 = bench_now();
            ds4_engine_compute_experts_batch(engine, (uint8_t)layer,
                xq_all, all_ids, all_wts, token_n,
                batch_size, n_experts, out_all);
            double t1 = bench_now();
            times[i] = (t1 - t0) * 1000.0;
            total += times[i];
        }

        /* Stats */
        double mean = total / n_iters;
        double min_t = times[0], max_t = times[0];
        double var = 0.0;
        for (int i = 0; i < n_iters; i++) {
            if (times[i] < min_t) min_t = times[i];
            if (times[i] > max_t) max_t = times[i];
            var += (times[i] - mean) * (times[i] - mean);
        }
        double stddev = sqrt(var / n_iters);

        for (int i = 0; i < n_iters - 1; i++)
            for (int j = i + 1; j < n_iters; j++)
                if (times[j] < times[i]) { double tmp = times[i]; times[i] = times[j]; times[j] = tmp; }
        double p50 = times[n_iters / 2];
        double p95 = times[(int)(n_iters * 0.95)];
        double p99 = times[(int)(n_iters * 0.99)];

        double mean_per_token = mean / batch_size;
        double tok_per_sec = (double)batch_size * 1000.0 / mean;

        fprintf(stderr, "\n");
        fprintf(stderr, "=== Batched expert compute (layer %d, %d experts, batch=%d) ===\n",
                layer, n_experts, batch_size);
        fprintf(stderr, "  Mean/batch:     %8.3f ms\n", mean);
        fprintf(stderr, "  Mean/token:     %8.3f ms  (%.1f tok/s)\n", mean_per_token, tok_per_sec);
        fprintf(stderr, "  Median/batch:   %8.3f ms\n", p50);
        fprintf(stderr, "  Min/batch:      %8.3f ms\n", min_t);
        fprintf(stderr, "  Max/batch:      %8.3f ms\n", max_t);
        fprintf(stderr, "  Stddev:         %8.3f ms\n", stddev);
        fprintf(stderr, "  P95:            %8.3f ms\n", p95);
        fprintf(stderr, "  P99:            %8.3f ms\n", p99);
        fprintf(stderr, "\n");

        fprintf(stderr, "  Throughput projection (43 layers):\n");
        fprintf(stderr, "    Expert-only latency/batch: %8.3f ms\n", mean * 43);
        fprintf(stderr, "    Expert-only latency/token: %8.3f ms  (%.1f tok/s)\n",
                mean_per_token * 43, 1000.0 / (mean_per_token * 43));
        fprintf(stderr, "\n");

        free(times);
        free(xq_all);
        free(out_all);
        free(all_ids);
        free(all_wts);
        free(token_n);
    }

    ds4_engine_close(engine);
    return 0;
}
