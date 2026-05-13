/*
 * ds4_expert_server.c — CPU-only expert shard server.
 *
 * Loads the GGUF model via mmap, listens on a TCP port, and serves MoE expert
 * computation requests from a coordinator running the attention/routing graph.
 *
 * Usage:
 *   ds4-expert-server <model.gguf> --port 9081 --experts 0-128
 *
 * The server only touches expert weight tensors in the mmap — attention,
 * embedding, and output weights are never read.  The CPU thread pool from
 * ds4.c handles the matmul parallelism.
 */

#include "ds4.h"
#include "ds4_moe_shard.h"

#include <arpa/inet.h>
#include <errno.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <pthread.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/uio.h>
#include <time.h>
#include <unistd.h>

static volatile sig_atomic_t g_stop = 0;

static void stop_handler(int sig) {
    (void)sig;
    g_stop = 1;
}

static double server_now_sec(void) {
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0) return 0.0;
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static int server_profile_enabled(void) {
    static int cached = -1;
    if (cached < 0) cached = (getenv("DS4_EXPERT_SHARD_PROFILE") != NULL);
    return cached;
}

static int full_send(int fd, const void *buf, size_t len) {
    const uint8_t *p = buf;
    while (len > 0) {
        ssize_t n = send(fd, p, len, 0);
        if (n < 0) {
            if (errno == EINTR && !g_stop) continue;
            return -1;
        }
        p += n;
        len -= (size_t)n;
    }
    return 0;
}

static int full_recv(int fd, void *buf, size_t len) {
    uint8_t *p = buf;
    while (len > 0) {
        ssize_t n = recv(fd, p, len, 0);
        if (n <= 0) {
            if (n < 0 && errno == EINTR && !g_stop) continue;
            return -1;
        }
        p += n;
        len -= (size_t)n;
    }
    return 0;
}

static int full_writev(int fd, struct iovec *iov, int iovcnt) {
    while (iovcnt > 0) {
        ssize_t n = writev(fd, iov, iovcnt);
        if (n < 0) {
            if (errno == EINTR && !g_stop) continue;
            return -1;
        }
        while (n > 0 && iovcnt > 0) {
            if ((size_t)n >= iov[0].iov_len) {
                n -= (ssize_t)iov[0].iov_len;
                iov++;
                iovcnt--;
            } else {
                iov[0].iov_base = (uint8_t *)iov[0].iov_base + n;
                iov[0].iov_len -= (size_t)n;
                n = 0;
            }
        }
    }
    return 0;
}

static void send_response(int fd, uint8_t status, uint8_t layer,
                          const float *out) {
    uint8_t hdr[DS4_SHARD_RSP_HDR_SIZE];
    uint32_t magic = DS4_SHARD_MAGIC;
    memcpy(hdr, &magic, 4);
    hdr[4] = status;
    hdr[5] = layer;
    hdr[6] = 0;
    hdr[7] = 0;
    if (status == DS4_SHARD_STATUS_OK && out) {
        struct iovec iov[2] = {
            { .iov_base = hdr, .iov_len = sizeof(hdr) },
            { .iov_base = (void *)out,
              .iov_len = (size_t)DS4_SHARD_N_EMBD * sizeof(float) },
        };
        full_writev(fd, iov, 2);
    } else {
        full_send(fd, hdr, sizeof(hdr));
    }
}

static void send_batch_response(int fd, uint8_t status, uint8_t layer,
                                uint16_t n_tokens, const float *out) {
    uint8_t hdr[DS4_SHARD_BATCH_RSP_HDR_SIZE];
    uint32_t magic = DS4_SHARD_MAGIC;
    memcpy(hdr, &magic, 4);
    hdr[4] = status;
    hdr[5] = layer;
    hdr[6] = 0;
    hdr[7] = 0;
    memcpy(hdr + 8, &n_tokens, sizeof(n_tokens));
    hdr[10] = 0;
    hdr[11] = 0;
    if (status == DS4_SHARD_STATUS_OK && out && n_tokens > 0) {
        struct iovec iov[2] = {
            { .iov_base = hdr, .iov_len = sizeof(hdr) },
            { .iov_base = (void *)out,
              .iov_len = (size_t)n_tokens * (size_t)DS4_SHARD_N_EMBD * sizeof(float) },
        };
        full_writev(fd, iov, 2);
    } else {
        full_send(fd, hdr, sizeof(hdr));
    }
}

static void handle_connection(int cfd, ds4_engine *engine,
                              uint16_t expert_start, uint16_t expert_end) {
    uint8_t hdr[DS4_SHARD_REQ_HDR_SIZE];
    uint16_t expert_ids[DS4_SHARD_N_EXPERT_USED];
    float expert_weights[DS4_SHARD_N_EXPERT_USED];
    uint8_t xq_buf[DS4_SHARD_Q8K_BYTES];
    float out[DS4_SHARD_N_EMBD];

    /* Pre-allocated batch buffers — grown as needed, reused across requests. */
    size_t batch_cap = 0;         /* current capacity in tokens */
    uint8_t *batch_token_n = NULL;
    uint16_t *batch_ids = NULL;
    float *batch_wts = NULL;
    uint8_t *batch_xq = NULL;
    float *batch_out = NULL;
    uint8_t *batch_recv_buf = NULL;
    size_t batch_recv_cap = 0;

    /* Cache profile setting once per connection (avoid getenv on hot path). */
    const int profile = server_profile_enabled();

    while (!g_stop) {
        const double t0 = profile ? server_now_sec() : 0.0;
        if (full_recv(cfd, hdr, sizeof(hdr)) != 0) break;

        uint32_t magic;
        memcpy(&magic, hdr, 4);
        if (magic != DS4_SHARD_MAGIC) {
            fprintf(stderr, "expert-server: bad magic from client\n");
            break;
        }

        uint8_t cmd = hdr[4];

        if (cmd == DS4_SHARD_CMD_PING) {
            send_response(cfd, DS4_SHARD_STATUS_OK, 0, NULL);
            continue;
        }

        if (cmd == DS4_SHARD_CMD_SHUTDOWN) {
            send_response(cfd, DS4_SHARD_STATUS_OK, 0, NULL);
            g_stop = 1;
            break;
        }

        if (cmd != DS4_SHARD_CMD_EXPERT) {
            if (cmd == DS4_SHARD_CMD_EXPERT_BATCH) {
                uint8_t tail[4];
                if (full_recv(cfd, tail, sizeof(tail)) != 0) break;
                uint16_t n_tokens = 0;
                uint16_t n_selected = 0;
                memcpy(&n_tokens, tail, sizeof(n_tokens));
                memcpy(&n_selected, tail + 2, sizeof(n_selected));
                uint8_t layer = hdr[5];
                uint8_t flags = hdr[6];
                if (layer >= DS4_SHARD_N_LAYER ||
                    n_tokens == 0 ||
                    n_selected == 0 ||
                    n_selected > DS4_SHARD_N_EXPERT_USED ||
                    !(flags & DS4_SHARD_FLAG_Q8K)) {
                    send_batch_response(cfd, DS4_SHARD_STATUS_ERR, layer, n_tokens, NULL);
                    continue;
                }

                /* Grow pre-allocated buffers if needed. */
                if ((size_t)n_tokens > batch_cap) {
                    size_t new_cap = (size_t)n_tokens;
                    size_t token_rows = new_cap * (size_t)DS4_SHARD_N_EXPERT_USED;
                    free(batch_token_n); batch_token_n = malloc(new_cap * sizeof(*batch_token_n));
                    free(batch_ids);     batch_ids     = malloc(token_rows * sizeof(*batch_ids));
                    free(batch_wts);     batch_wts     = malloc(token_rows * sizeof(*batch_wts));
                    free(batch_xq);      batch_xq      = malloc(new_cap * DS4_SHARD_Q8K_BYTES);
                    free(batch_out);     batch_out     = malloc(new_cap * (size_t)DS4_SHARD_N_EMBD * sizeof(*batch_out));
                    if (!batch_token_n || !batch_ids || !batch_wts || !batch_xq || !batch_out) {
                        send_batch_response(cfd, DS4_SHARD_STATUS_ERR, layer, n_tokens, NULL);
                        continue;
                    }
                    batch_cap = new_cap;
                }

                /* Coalesced recv: read all per-token data in one syscall. */
                const size_t per_token =
                    sizeof(uint32_t) +
                    (size_t)n_selected * sizeof(uint16_t) +
                    (size_t)n_selected * sizeof(float) +
                    DS4_SHARD_Q8K_BYTES;
                const size_t total_recv = (size_t)n_tokens * per_token;
                if (total_recv > batch_recv_cap) {
                    free(batch_recv_buf);
                    batch_recv_buf = malloc(total_recv);
                    if (!batch_recv_buf) {
                        batch_recv_cap = 0;
                        send_batch_response(cfd, DS4_SHARD_STATUS_ERR, layer, n_tokens, NULL);
                        continue;
                    }
                    batch_recv_cap = total_recv;
                }

                if (full_recv(cfd, batch_recv_buf, total_recv) != 0) break;
                const double t_recv = profile ? server_now_sec() : 0.0;

                /* Unpack the coalesced buffer into structured arrays. */
                bool ok_batch = true;
                const uint8_t *rp = batch_recv_buf;
                for (uint16_t t = 0; t < n_tokens && ok_batch; t++) {
                    uint32_t token_hdr = 0;
                    memcpy(&token_hdr, rp, sizeof(token_hdr));
                    rp += sizeof(token_hdr);
                    batch_token_n[t] = (uint8_t)(token_hdr & 0xFFu);
                    if (batch_token_n[t] > n_selected) { ok_batch = false; break; }

                    uint16_t *id_row = batch_ids + (size_t)t * (size_t)n_selected;
                    float *w_row = batch_wts + (size_t)t * (size_t)n_selected;
                    memcpy(id_row, rp, (size_t)n_selected * sizeof(uint16_t));
                    rp += (size_t)n_selected * sizeof(uint16_t);
                    memcpy(w_row, rp, (size_t)n_selected * sizeof(float));
                    rp += (size_t)n_selected * sizeof(float);
                    memcpy(batch_xq + (size_t)t * DS4_SHARD_Q8K_BYTES, rp,
                           DS4_SHARD_Q8K_BYTES);
                    rp += DS4_SHARD_Q8K_BYTES;

                    for (uint8_t i = 0; i < batch_token_n[t]; i++) {
                        if (id_row[i] < expert_start || id_row[i] >= expert_end) {
                            ok_batch = false;
                            break;
                        }
                    }
                }
                const double t_unpack = profile ? server_now_sec() : 0.0;

                /* Batched expert compute — single call. */
                if (ok_batch) {
                    ok_batch = (ds4_engine_compute_experts_batch(
                        engine, layer, batch_xq, batch_ids, batch_wts, batch_token_n,
                        (int)n_tokens, (int)n_selected, batch_out) == 0);
                }
                const double t_compute = profile ? server_now_sec() : 0.0;

                if (!ok_batch) {
                    send_batch_response(cfd, DS4_SHARD_STATUS_ERR, layer, n_tokens, NULL);
                } else {
                    send_batch_response(cfd, DS4_SHARD_STATUS_OK, layer, n_tokens, batch_out);
                }
                if (profile) {
                    const double t_done = server_now_sec();
                    fprintf(stderr,
                            "expert-server: batch layer=%u tokens=%u selected=%u "
                            "recv=%.3f unpack=%.3f compute=%.3f send=%.3f total=%.3f ms\n",
                            layer,
                            n_tokens,
                            n_selected,
                            (t_recv - t0) * 1000.0,
                            (t_unpack - t_recv) * 1000.0,
                            (t_compute - t_unpack) * 1000.0,
                            (t_done - t_compute) * 1000.0,
                            (t_done - t0) * 1000.0);
                }

                continue;
            }
            fprintf(stderr, "expert-server: unknown command 0x%02x\n", cmd);
            send_response(cfd, DS4_SHARD_STATUS_ERR, 0, NULL);
            break;
        }

        uint8_t layer = hdr[5];
        uint8_t n_experts = hdr[6];
        uint8_t flags = hdr[7];

        if (layer >= DS4_SHARD_N_LAYER || n_experts == 0 ||
            n_experts > DS4_SHARD_N_EXPERT_USED)
        {
            send_response(cfd, DS4_SHARD_STATUS_ERR, layer, NULL);
            continue;
        }

        if (!(flags & DS4_SHARD_FLAG_Q8K)) {
            send_response(cfd, DS4_SHARD_STATUS_ERR, layer, NULL);
            continue;
        }

        /* Coalesced recv: read ids + weights + activation in one call. */
        const size_t ids_sz = (size_t)n_experts * sizeof(uint16_t);
        const size_t wts_sz = (size_t)n_experts * sizeof(float);
        const size_t body_sz = ids_sz + wts_sz + DS4_SHARD_Q8K_BYTES;
        uint8_t recv_body[DS4_SHARD_N_EXPERT_USED * (sizeof(uint16_t) + sizeof(float)) + DS4_SHARD_Q8K_BYTES];
        if (full_recv(cfd, recv_body, body_sz) != 0) break;
        memcpy(expert_ids, recv_body, ids_sz);
        memcpy(expert_weights, recv_body + ids_sz, wts_sz);
        memcpy(xq_buf, recv_body + ids_sz + wts_sz, DS4_SHARD_Q8K_BYTES);
        const double t_recv = profile ? server_now_sec() : 0.0;

        /* Validate expert ownership. */
        bool valid = true;
        for (int i = 0; i < n_experts; i++) {
            if (expert_ids[i] < expert_start || expert_ids[i] >= expert_end) {
                fprintf(stderr, "expert-server: expert %u not in range %u–%u\n",
                        expert_ids[i], expert_start, expert_end);
                valid = false;
            }
        }
        if (!valid) {
            send_response(cfd, DS4_SHARD_STATUS_ERR, layer, NULL);
            continue;
        }

        /* Compute expert forward pass. */
        int rc = ds4_engine_compute_experts(engine, layer, xq_buf,
                                            expert_ids, expert_weights,
                                            n_experts, out);
        const double t_compute = profile ? server_now_sec() : 0.0;
        if (rc != 0) {
            send_response(cfd, DS4_SHARD_STATUS_ERR, layer, NULL);
            continue;
        }

        send_response(cfd, DS4_SHARD_STATUS_OK, layer, out);
        if (profile) {
            const double t_done = server_now_sec();
            fprintf(stderr,
                    "expert-server: profile layer=%u experts=%u recv=%.3f ms compute=%.3f ms send=%.3f ms total=%.3f ms\n",
                    layer,
                    n_experts,
                    (t_recv - t0) * 1000.0,
                    (t_compute - t_recv) * 1000.0,
                    (t_done - t_compute) * 1000.0,
                    (t_done - t0) * 1000.0);
        }
    }

    /* Free pre-allocated batch buffers. */
    free(batch_token_n);
    free(batch_ids);
    free(batch_wts);
    free(batch_xq);
    free(batch_out);
    free(batch_recv_buf);
}

static void usage(void) {
    fprintf(stderr,
        "Usage: ds4-expert-server <model.gguf> [options]\n"
        "\n"
        "Options:\n"
        "  --port N          TCP port to listen on (default: 9081)\n"
        "  --experts S-E     Expert range start-end exclusive (e.g. 0-128)\n"
        "  --threads N       CPU threads for matmul (default: auto)\n"
        "\n");
    exit(1);
}

int main(int argc, char **argv) {
    const char *model_path = NULL;
    uint16_t port = 9081;
    uint16_t expert_start = 0;
    uint16_t expert_end = DS4_SHARD_N_EXPERT;
    int n_threads = 0;

    for (int i = 1; i < argc; i++) {
        if (argv[i][0] != '-' && !model_path) {
            model_path = argv[i];
        } else if (strcmp(argv[i], "--port") == 0 && i + 1 < argc) {
            port = (uint16_t)atoi(argv[++i]);
        } else if (strcmp(argv[i], "--experts") == 0 && i + 1 < argc) {
            char *dash = strchr(argv[++i], '-');
            if (!dash) { fprintf(stderr, "bad --experts format\n"); usage(); }
            *dash = '\0';
            expert_start = (uint16_t)atoi(argv[i]);
            expert_end = (uint16_t)atoi(dash + 1);
        } else if (strcmp(argv[i], "--threads") == 0 && i + 1 < argc) {
            n_threads = atoi(argv[++i]);
        } else {
            usage();
        }
    }

    if (!model_path) usage();
    if (expert_start >= expert_end || expert_end > DS4_SHARD_N_EXPERT) {
        fprintf(stderr, "invalid expert range %u–%u\n", expert_start, expert_end);
        return 1;
    }

    fprintf(stderr, "ds4-expert-server: loading %s (experts %u–%u)\n",
            model_path, expert_start, expert_end);

    /* Load model with CPU backend in expert-only mode — skip GPU, release
     * non-expert pages, keep only expert weight tensors resident. */
    ds4_engine_options opts = {
        .model_path = model_path,
        .backend = DS4_BACKEND_CPU,
        .n_threads = n_threads,
        .expert_only = true,
    };
    ds4_engine *engine = NULL;
    if (ds4_engine_open(&engine, &opts) != 0) {
        fprintf(stderr, "ds4-expert-server: failed to load model\n");
        return 1;
    }

    /* Set up listening socket. */
    int lfd = socket(AF_INET6, SOCK_STREAM, 0);
    if (lfd < 0) { perror("socket"); return 1; }

    int opt = 1;
    setsockopt(lfd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    struct sockaddr_in6 addr = {
        .sin6_family = AF_INET6,
        .sin6_port = htons(port),
        .sin6_addr = in6addr_any,
    };
    if (bind(lfd, (struct sockaddr *)&addr, sizeof(addr)) != 0) {
        perror("bind");
        close(lfd);
        return 1;
    }
    if (listen(lfd, 8) != 0) {
        perror("listen");
        close(lfd);
        return 1;
    }

    /* Use sigaction without SA_RESTART so that blocking recv/send/accept
     * return EINTR when the signal fires — allows clean shutdown. */
    struct sigaction sa = { .sa_handler = stop_handler };
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = 0;  /* no SA_RESTART */
    sigaction(SIGINT, &sa, NULL);
    sigaction(SIGTERM, &sa, NULL);
    signal(SIGPIPE, SIG_IGN);

    fprintf(stderr, "ds4-expert-server: listening on port %u\n", port);

    while (!g_stop) {
        struct sockaddr_storage client_addr;
        socklen_t client_len = sizeof(client_addr);
        int cfd = accept(lfd, (struct sockaddr *)&client_addr, &client_len);
        if (cfd < 0) {
            if (errno == EINTR) continue;
            break;
        }

        int flag = 1;
        int bufsize = 4 * 1024 * 1024;
        setsockopt(cfd, IPPROTO_TCP, TCP_NODELAY, &flag, sizeof(flag));
        setsockopt(cfd, SOL_SOCKET, SO_SNDBUF, &bufsize, sizeof(bufsize));
        setsockopt(cfd, SOL_SOCKET, SO_RCVBUF, &bufsize, sizeof(bufsize));

        fprintf(stderr, "ds4-expert-server: client connected\n");
        handle_connection(cfd, engine, expert_start, expert_end);
        close(cfd);
        fprintf(stderr, "ds4-expert-server: client disconnected\n");
    }

    close(lfd);
    ds4_engine_close(engine);
    fprintf(stderr, "ds4-expert-server: shutdown\n");
    return 0;
}
