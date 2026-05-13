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
#include <unistd.h>

static volatile sig_atomic_t g_stop = 0;

static void stop_handler(int sig) {
    (void)sig;
    g_stop = 1;
}

static int full_send(int fd, const void *buf, size_t len) {
    const uint8_t *p = buf;
    while (len > 0) {
        ssize_t n = send(fd, p, len, 0);
        if (n < 0) {
            if (errno == EINTR) continue;
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
            if (n < 0 && errno == EINTR) continue;
            return -1;
        }
        p += n;
        len -= (size_t)n;
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
    full_send(fd, hdr, sizeof(hdr));
    if (status == DS4_SHARD_STATUS_OK && out) {
        full_send(fd, out, (size_t)DS4_SHARD_N_EMBD * sizeof(float));
    }
}

static void handle_connection(int cfd, ds4_engine *engine,
                              uint16_t expert_start, uint16_t expert_end) {
    uint8_t hdr[DS4_SHARD_REQ_HDR_SIZE];
    uint16_t expert_ids[DS4_SHARD_N_EXPERT_USED];
    float expert_weights[DS4_SHARD_N_EXPERT_USED];
    uint8_t xq_buf[DS4_SHARD_Q8K_BYTES];
    float out[DS4_SHARD_N_EMBD];

    while (!g_stop) {
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

        if (full_recv(cfd, expert_ids, (size_t)n_experts * sizeof(uint16_t)) != 0) break;
        if (full_recv(cfd, expert_weights, (size_t)n_experts * sizeof(float)) != 0) break;

        size_t act_size = (flags & DS4_SHARD_FLAG_Q8K) ?
                          DS4_SHARD_Q8K_BYTES :
                          (size_t)DS4_SHARD_N_EMBD * sizeof(float);
        if (act_size > sizeof(xq_buf)) {
            /* f32 activation is larger; for now only support Q8_K. */
            send_response(cfd, DS4_SHARD_STATUS_ERR, layer, NULL);
            continue;
        }
        if (full_recv(cfd, xq_buf, act_size) != 0) break;

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
        if (rc != 0) {
            send_response(cfd, DS4_SHARD_STATUS_ERR, layer, NULL);
            continue;
        }

        send_response(cfd, DS4_SHARD_STATUS_OK, layer, out);
    }
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

    signal(SIGINT, stop_handler);
    signal(SIGTERM, stop_handler);
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
        setsockopt(cfd, IPPROTO_TCP, TCP_NODELAY, &flag, sizeof(flag));

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
