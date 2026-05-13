/*
 * ds4_moe_shard.c — MoE expert shard client.
 *
 * Persistent TCP connections to remote expert servers.  Each shard owns a
 * contiguous range of expert IDs and runs gate/up/SwiGLU/down on CPU.
 * The pool fans out one layer at a time using per-shard worker pthreads.
 */

#include "ds4_moe_shard.h"

#include <arpa/inet.h>
#include <errno.h>
#include <netdb.h>
#include <netinet/tcp.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <time.h>
#include <unistd.h>

/* ---- helpers ----------------------------------------------------------- */

static void die(const char *msg) {
    fprintf(stderr, "ds4_moe_shard: %s\n", msg);
    exit(1);
}

static double shard_now_sec(void) {
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0) return 0.0;
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static int shard_profile_enabled(void) {
    return getenv("DS4_EXPERT_SHARD_PROFILE") != NULL;
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

/* ---- single shard ------------------------------------------------------ */

struct ds4_shard {
    int fd;
    char host[256];
    uint16_t port;
    uint16_t expert_start;
    uint16_t expert_end;
};

static int shard_tcp_connect(const char *host, uint16_t port) {
    char port_str[8];
    snprintf(port_str, sizeof(port_str), "%u", port);

    struct addrinfo hints = { .ai_family = AF_UNSPEC, .ai_socktype = SOCK_STREAM };
    struct addrinfo *res = NULL;
    if (getaddrinfo(host, port_str, &hints, &res) != 0 || !res) return -1;

    int fd = socket(res->ai_family, res->ai_socktype, res->ai_protocol);
    if (fd < 0) { freeaddrinfo(res); return -1; }

    if (connect(fd, res->ai_addr, res->ai_addrlen) != 0) {
        close(fd);
        freeaddrinfo(res);
        return -1;
    }
    freeaddrinfo(res);

    int flag = 1;
    setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &flag, sizeof(flag));
    return fd;
}

int ds4_shard_connect(ds4_shard **out, const ds4_shard_config *cfg) {
    ds4_shard *s = calloc(1, sizeof(*s));
    if (!s) return -1;

    snprintf(s->host, sizeof(s->host), "%s", cfg->host);
    s->port = cfg->port;
    s->expert_start = cfg->expert_start;
    s->expert_end = cfg->expert_end;

    s->fd = shard_tcp_connect(cfg->host, cfg->port);
    if (s->fd < 0) {
        fprintf(stderr, "ds4_moe_shard: failed to connect to %s:%u\n",
                cfg->host, cfg->port);
        free(s);
        return -1;
    }

    fprintf(stderr, "ds4_moe_shard: connected to %s:%u (experts %u–%u)\n",
            cfg->host, cfg->port, cfg->expert_start, cfg->expert_end);
    *out = s;
    return 0;
}

void ds4_shard_close(ds4_shard *s) {
    if (!s) return;
    if (s->fd >= 0) close(s->fd);
    free(s);
}

bool ds4_shard_owns(const ds4_shard *s, int expert_id) {
    return expert_id >= s->expert_start && expert_id < s->expert_end;
}

bool ds4_shard_ping(ds4_shard *s) {
    uint8_t req[DS4_SHARD_REQ_HDR_SIZE];
    uint32_t magic = DS4_SHARD_MAGIC;
    memcpy(req, &magic, 4);
    req[4] = DS4_SHARD_CMD_PING;
    req[5] = 0;
    req[6] = 0;
    req[7] = 0;
    if (full_send(s->fd, req, sizeof(req)) != 0) return false;

    uint8_t rsp[DS4_SHARD_RSP_HDR_SIZE];
    if (full_recv(s->fd, rsp, sizeof(rsp)) != 0) return false;

    uint32_t rsp_magic;
    memcpy(&rsp_magic, rsp, 4);
    return rsp_magic == DS4_SHARD_MAGIC && rsp[4] == DS4_SHARD_STATUS_OK;
}

int ds4_shard_dispatch_layer(
    ds4_shard        *s,
    uint8_t           layer,
    const void       *xq,
    const uint16_t   *expert_ids,
    const float      *expert_weights,
    int               n_experts,
    float            *out)
{
    if (n_experts <= 0 || n_experts > DS4_SHARD_N_EXPERT_USED) return -1;
    const int profile = shard_profile_enabled();
    const double t0 = profile ? shard_now_sec() : 0.0;

    /* Build request. */
    uint8_t hdr[DS4_SHARD_REQ_HDR_SIZE];
    uint32_t magic = DS4_SHARD_MAGIC;
    memcpy(hdr, &magic, 4);
    hdr[4] = DS4_SHARD_CMD_EXPERT;
    hdr[5] = layer;
    hdr[6] = (uint8_t)n_experts;
    hdr[7] = DS4_SHARD_FLAG_Q8K;

    if (full_send(s->fd, hdr, sizeof(hdr)) != 0) return -1;
    if (full_send(s->fd, expert_ids, (size_t)n_experts * sizeof(uint16_t)) != 0) return -1;
    if (full_send(s->fd, expert_weights, (size_t)n_experts * sizeof(float)) != 0) return -1;
    if (full_send(s->fd, xq, DS4_SHARD_Q8K_BYTES) != 0) return -1;
    const double t_sent = profile ? shard_now_sec() : 0.0;

    /* Read response. */
    uint8_t rsp_hdr[DS4_SHARD_RSP_HDR_SIZE];
    if (full_recv(s->fd, rsp_hdr, sizeof(rsp_hdr)) != 0) return -1;
    const double t_rsp_hdr = profile ? shard_now_sec() : 0.0;

    uint32_t rsp_magic;
    memcpy(&rsp_magic, rsp_hdr, 4);
    if (rsp_magic != DS4_SHARD_MAGIC) return -1;
    if (rsp_hdr[4] != DS4_SHARD_STATUS_OK) return -1;

    if (full_recv(s->fd, out, (size_t)DS4_SHARD_N_EMBD * sizeof(float)) != 0) return -1;
    if (profile) {
        const double t_done = shard_now_sec();
        fprintf(stderr,
                "ds4_moe_shard: profile host=%s:%u layer=%u experts=%d send=%.3f ms wait_hdr=%.3f ms recv_out=%.3f ms total=%.3f ms\n",
                s->host,
                s->port,
                layer,
                n_experts,
                (t_sent - t0) * 1000.0,
                (t_rsp_hdr - t_sent) * 1000.0,
                (t_done - t_rsp_hdr) * 1000.0,
                (t_done - t0) * 1000.0);
    }
    return 0;
}

/* ---- shard pool: parallel fan-out -------------------------------------- */

typedef struct {
    ds4_shard *shard;

    /* Per-dispatch work (set by main thread before signal). */
    uint8_t   layer;
    const void *xq;
    uint16_t  expert_ids[DS4_SHARD_N_EXPERT_USED];
    float     expert_weights[DS4_SHARD_N_EXPERT_USED];
    int       n_experts;
    float     out[DS4_SHARD_N_EMBD];
    int       result;

    /* Synchronization. */
    pthread_t       thread;
    pthread_mutex_t mutex;
    pthread_cond_t  cond;
    bool            has_work;
    bool            shutdown;
    bool            done;
} shard_worker;

struct ds4_shard_pool {
    shard_worker *workers;
    int           n_shards;
    pthread_mutex_t done_mutex;
    pthread_cond_t  done_cond;
};

static void *shard_worker_main(void *arg) {
    shard_worker *w = arg;

    for (;;) {
        pthread_mutex_lock(&w->mutex);
        while (!w->has_work && !w->shutdown)
            pthread_cond_wait(&w->cond, &w->mutex);

        if (w->shutdown) {
            pthread_mutex_unlock(&w->mutex);
            break;
        }

        /* Do the dispatch. */
        w->has_work = false;
        pthread_mutex_unlock(&w->mutex);

        if (w->n_experts > 0) {
            w->result = ds4_shard_dispatch_layer(
                w->shard, w->layer, w->xq,
                w->expert_ids, w->expert_weights, w->n_experts,
                w->out);
        } else {
            w->result = 0;
            memset(w->out, 0, sizeof(w->out));
        }

        /* Signal completion. */
        pthread_mutex_lock(&w->mutex);
        w->done = true;
        pthread_cond_signal(&w->cond);
        pthread_mutex_unlock(&w->mutex);
    }
    return NULL;
}

int ds4_shard_pool_create(ds4_shard_pool **out,
                          const ds4_shard_config *cfgs, int n_shards) {
    if (n_shards <= 0 || n_shards > 32) return -1;

    ds4_shard_pool *p = calloc(1, sizeof(*p));
    if (!p) return -1;
    p->n_shards = n_shards;
    p->workers = calloc((size_t)n_shards, sizeof(shard_worker));
    if (!p->workers) { free(p); return -1; }
    pthread_mutex_init(&p->done_mutex, NULL);
    pthread_cond_init(&p->done_cond, NULL);

    for (int i = 0; i < n_shards; i++) {
        shard_worker *w = &p->workers[i];
        if (ds4_shard_connect(&w->shard, &cfgs[i]) != 0) {
            /* Clean up already-connected shards. */
            for (int j = 0; j < i; j++) {
                p->workers[j].shutdown = true;
                pthread_cond_signal(&p->workers[j].cond);
                pthread_join(p->workers[j].thread, NULL);
                pthread_mutex_destroy(&p->workers[j].mutex);
                pthread_cond_destroy(&p->workers[j].cond);
                ds4_shard_close(p->workers[j].shard);
            }
            free(p->workers);
            free(p);
            return -1;
        }

        pthread_mutex_init(&w->mutex, NULL);
        pthread_cond_init(&w->cond, NULL);
        w->has_work = false;
        w->shutdown = false;
        w->done = false;

        if (pthread_create(&w->thread, NULL, shard_worker_main, w) != 0) {
            die("failed to create shard worker thread");
        }
    }

    /* Ping all shards to verify connectivity. */
    for (int i = 0; i < n_shards; i++) {
        if (!ds4_shard_ping(p->workers[i].shard)) {
            fprintf(stderr, "ds4_moe_shard: ping failed for shard %d (%s:%u)\n",
                    i, cfgs[i].host, cfgs[i].port);
            ds4_shard_pool_close(p);
            return -1;
        }
    }

    fprintf(stderr, "ds4_moe_shard: pool ready with %d shard(s)\n", n_shards);
    *out = p;
    return 0;
}

void ds4_shard_pool_close(ds4_shard_pool *p) {
    if (!p) return;
    for (int i = 0; i < p->n_shards; i++) {
        shard_worker *w = &p->workers[i];
        pthread_mutex_lock(&w->mutex);
        w->shutdown = true;
        pthread_cond_signal(&w->cond);
        pthread_mutex_unlock(&w->mutex);
        pthread_join(w->thread, NULL);
        pthread_mutex_destroy(&w->mutex);
        pthread_cond_destroy(&w->cond);
        ds4_shard_close(w->shard);
    }
    pthread_mutex_destroy(&p->done_mutex);
    pthread_cond_destroy(&p->done_cond);
    free(p->workers);
    free(p);
}

int ds4_shard_pool_dispatch_layer(
    ds4_shard_pool   *p,
    uint8_t           layer,
    const void       *xq,
    const int        *selected,
    const float      *weights,
    int               n_selected,
    float            *out)
{
    const int profile = shard_profile_enabled();
    const double t0 = profile ? shard_now_sec() : 0.0;
    /* Partition selected experts by shard ownership and kick off workers. */
    const double t_signaled = profile ? shard_now_sec() : 0.0;

    for (int si = 0; si < p->n_shards; si++) {
        shard_worker *w = &p->workers[si];
        w->layer = layer;
        w->xq = xq;
        w->n_experts = 0;

        for (int ei = 0; ei < n_selected; ei++) {
            if (ds4_shard_owns(w->shard, selected[ei])) {
                int k = w->n_experts;
                w->expert_ids[k] = (uint16_t)selected[ei];
                w->expert_weights[k] = weights[ei];
                w->n_experts++;
            }
        }

        pthread_mutex_lock(&w->mutex);
        w->done = false;
        w->has_work = true;
        pthread_cond_signal(&w->cond);
        pthread_mutex_unlock(&w->mutex);
    }

    /* Wait for all workers and sum partial results. */
    memset(out, 0, (size_t)DS4_SHARD_N_EMBD * sizeof(float));

    for (int si = 0; si < p->n_shards; si++) {
        shard_worker *w = &p->workers[si];
        pthread_mutex_lock(&w->mutex);
        while (!w->done)
            pthread_cond_wait(&w->cond, &w->mutex);
        pthread_mutex_unlock(&w->mutex);

        if (w->result != 0) {
            fprintf(stderr, "ds4_moe_shard: dispatch failed for shard %d layer %u\n",
                    si, layer);
            return -1;
        }

        if (w->n_experts > 0) {
            for (int d = 0; d < DS4_SHARD_N_EMBD; d++)
                out[d] += w->out[d];
        }
    }

    if (profile) {
        const double t_done = shard_now_sec();
        fprintf(stderr,
                "ds4_moe_shard: pool profile layer=%u shards=%d selected=%d signal=%.3f ms wait_sum=%.3f ms total=%.3f ms\n",
                layer,
                p->n_shards,
                n_selected,
                (t_signaled - t0) * 1000.0,
                (t_done - t_signaled) * 1000.0,
                (t_done - t0) * 1000.0);
    }
    return 0;
}

/* ---- config parser ----------------------------------------------------- */

int ds4_shard_config_parse(const char *spec, ds4_shard_config **out) {
    if (!spec || !spec[0]) return -1;

    /* Count commas to estimate number of shards. */
    int n = 1;
    for (const char *p = spec; *p; p++) {
        if (*p == ',') n++;
    }

    ds4_shard_config *cfgs = calloc((size_t)n, sizeof(ds4_shard_config));
    if (!cfgs) return -1;

    char *dup = strdup(spec);
    if (!dup) { free(cfgs); return -1; }

    int count = 0;
    char *saveptr = NULL;
    char *tok = strtok_r(dup, ",", &saveptr);
    while (tok && count < n) {
        /* Expected format: "host:port:start-end" */
        char *host_end = strrchr(tok, ':');
        if (!host_end) goto parse_err;
        *host_end = '\0';
        char *range_str = host_end + 1;

        char *port_end = strrchr(tok, ':');
        if (!port_end) goto parse_err;
        *port_end = '\0';
        char *port_str = port_end + 1;

        const char *host = tok;
        unsigned long port = strtoul(port_str, NULL, 10);
        if (port == 0 || port > 65535) goto parse_err;

        char *dash = strchr(range_str, '-');
        if (!dash) goto parse_err;
        *dash = '\0';
        unsigned long start = strtoul(range_str, NULL, 10);
        unsigned long end = strtoul(dash + 1, NULL, 10);

        /* Allocate host string — pool_create will copy via shard_connect. */
        cfgs[count].host = strdup(host);
        cfgs[count].port = (uint16_t)port;
        cfgs[count].expert_start = (uint16_t)start;
        cfgs[count].expert_end = (uint16_t)end;
        count++;

        tok = strtok_r(NULL, ",", &saveptr);
    }

    free(dup);
    *out = cfgs;
    return count;

parse_err:
    free(dup);
    for (int i = 0; i < count; i++) free((char *)cfgs[i].host);
    free(cfgs);
    return -1;
}
