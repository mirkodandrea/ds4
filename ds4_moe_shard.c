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
    static int cached = -1;
    if (cached < 0) cached = (getenv("DS4_EXPERT_SHARD_PROFILE") != NULL);
    return cached;
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
    /* Persistent send buffer — grows only, reused across dispatches. */
    uint8_t *sendbuf;
    size_t   sendbuf_cap;
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
    int bufsize = 4 * 1024 * 1024;
    setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &flag, sizeof(flag));
    setsockopt(fd, SOL_SOCKET, SO_SNDBUF, &bufsize, sizeof(bufsize));
    setsockopt(fd, SOL_SOCKET, SO_RCVBUF, &bufsize, sizeof(bufsize));
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
    free(s->sendbuf);
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

    /* Build and send request in one coalesced write. */
    uint8_t hdr[DS4_SHARD_REQ_HDR_SIZE];
    uint32_t magic = DS4_SHARD_MAGIC;
    memcpy(hdr, &magic, 4);
    hdr[4] = DS4_SHARD_CMD_EXPERT;
    hdr[5] = layer;
    hdr[6] = (uint8_t)n_experts;
    hdr[7] = DS4_SHARD_FLAG_Q8K;

    const size_t ids_sz = (size_t)n_experts * sizeof(uint16_t);
    const size_t wts_sz = (size_t)n_experts * sizeof(float);
    const size_t total = sizeof(hdr) + ids_sz + wts_sz + DS4_SHARD_Q8K_BYTES;
    uint8_t sendbuf_stack[DS4_SHARD_REQ_HDR_SIZE + DS4_SHARD_N_EXPERT_USED * (sizeof(uint16_t) + sizeof(float)) + DS4_SHARD_Q8K_BYTES];
    uint8_t *buf = (total <= sizeof(sendbuf_stack)) ? sendbuf_stack : malloc(total);
    if (!buf) return -1;

    memcpy(buf, hdr, sizeof(hdr));
    memcpy(buf + sizeof(hdr), expert_ids, ids_sz);
    memcpy(buf + sizeof(hdr) + ids_sz, expert_weights, wts_sz);
    memcpy(buf + sizeof(hdr) + ids_sz + wts_sz, xq, DS4_SHARD_Q8K_BYTES);
    int send_rc = full_send(s->fd, buf, total);
    if (buf != sendbuf_stack) free(buf);
    if (send_rc != 0) return -1;
    const double t_sent = profile ? shard_now_sec() : 0.0;

    /* Read response header, then output data directly into caller's buffer. */
    uint8_t rsp_hdr[DS4_SHARD_RSP_HDR_SIZE];
    if (full_recv(s->fd, rsp_hdr, sizeof(rsp_hdr)) != 0) return -1;

    uint32_t rsp_magic;
    memcpy(&rsp_magic, rsp_hdr, 4);
    if (rsp_magic != DS4_SHARD_MAGIC) return -1;
    if (rsp_hdr[4] != DS4_SHARD_STATUS_OK) return -1;

    if (full_recv(s->fd, out, (size_t)DS4_SHARD_N_EMBD * sizeof(float)) != 0) return -1;
    if (profile) {
        const double t_done = shard_now_sec();
        fprintf(stderr,
                "ds4_moe_shard: profile host=%s:%u layer=%u experts=%d send=%.3f ms recv=%.3f ms total=%.3f ms\n",
                s->host,
                s->port,
                layer,
                n_experts,
                (t_sent - t0) * 1000.0,
                (t_done - t_sent) * 1000.0,
                (t_done - t0) * 1000.0);
    }
    return 0;
}

/* Batch variant: one request computes one layer for n_tokens activations.
 * expert_ids/expert_weights are fixed-width rows: n_tokens × n_selected.
 * token_n_experts[t] indicates how many entries in each row are valid. */
static int ds4_shard_dispatch_layer_batch(
    ds4_shard        *s,
    uint8_t           layer,
    const void       *xq_batch,        /* n_tokens × DS4_SHARD_Q8K_BYTES */
    const uint16_t   *expert_ids,      /* n_tokens × n_selected */
    const float      *expert_weights,  /* n_tokens × n_selected */
    const uint8_t    *token_n_experts, /* n_tokens */
    int               n_tokens,
    int               n_selected,
    float            *out_batch)       /* n_tokens × DS4_SHARD_N_EMBD */
{
    if (n_tokens <= 0 || n_selected <= 0 || n_selected > DS4_SHARD_N_EXPERT_USED) return -1;
    const int profile = shard_profile_enabled();
    const double t0 = profile ? shard_now_sec() : 0.0;

    uint8_t hdr[DS4_SHARD_BATCH_REQ_HDR_SIZE];
    uint32_t magic = DS4_SHARD_MAGIC;
    memcpy(hdr, &magic, 4);
    hdr[4] = DS4_SHARD_CMD_EXPERT_BATCH;
    hdr[5] = layer;
    hdr[6] = DS4_SHARD_FLAG_Q8K;
    hdr[7] = 0;
    uint16_t n_tok_u16 = (uint16_t)n_tokens;
    uint16_t n_sel_u16 = (uint16_t)n_selected;
    memcpy(hdr + 8, &n_tok_u16, sizeof(n_tok_u16));
    memcpy(hdr + 10, &n_sel_u16, sizeof(n_sel_u16));

    /* Coalesce header + all per-token data into one contiguous send to
     * minimize syscall overhead (was 4 sends × n_tokens before). */
    const size_t per_token =
        sizeof(uint32_t) +                              /* token_hdr */
        (size_t)n_selected * sizeof(uint16_t) +         /* ids */
        (size_t)n_selected * sizeof(float) +            /* weights */
        DS4_SHARD_Q8K_BYTES;                            /* activation */
    const size_t total_send = sizeof(hdr) + (size_t)n_tokens * per_token;
    if (total_send > s->sendbuf_cap) {
        free(s->sendbuf);
        s->sendbuf = malloc(total_send);
        if (!s->sendbuf) {
            s->sendbuf_cap = 0;
            return -1;
        }
        s->sendbuf_cap = total_send;
    }
    uint8_t *sendbuf = s->sendbuf;

    memcpy(sendbuf, hdr, sizeof(hdr));
    uint8_t *wp = sendbuf + sizeof(hdr);
    for (int t = 0; t < n_tokens; t++) {
        const uint32_t token_hdr = ((uint32_t)token_n_experts[t] & 0xFFu);
        memcpy(wp, &token_hdr, sizeof(token_hdr));
        wp += sizeof(token_hdr);
        memcpy(wp, expert_ids + (size_t)t * (size_t)n_selected,
               (size_t)n_selected * sizeof(uint16_t));
        wp += (size_t)n_selected * sizeof(uint16_t);
        memcpy(wp, expert_weights + (size_t)t * (size_t)n_selected,
               (size_t)n_selected * sizeof(float));
        wp += (size_t)n_selected * sizeof(float);
        memcpy(wp, (const uint8_t *)xq_batch + (size_t)t * DS4_SHARD_Q8K_BYTES,
               DS4_SHARD_Q8K_BYTES);
        wp += DS4_SHARD_Q8K_BYTES;
    }

    int send_rc = full_send(s->fd, sendbuf, total_send);
    if (send_rc != 0) return -1;
    const double t_sent = profile ? shard_now_sec() : 0.0;

    uint8_t rsp_hdr[DS4_SHARD_BATCH_RSP_HDR_SIZE];
    if (full_recv(s->fd, rsp_hdr, sizeof(rsp_hdr)) != 0) return -1;
    const double t_rsp_hdr = profile ? shard_now_sec() : 0.0;

    uint32_t rsp_magic;
    memcpy(&rsp_magic, rsp_hdr, 4);
    if (rsp_magic != DS4_SHARD_MAGIC) return -1;
    if (rsp_hdr[4] != DS4_SHARD_STATUS_OK) return -1;
    uint16_t rsp_n_tok = 0;
    memcpy(&rsp_n_tok, rsp_hdr + 8, sizeof(rsp_n_tok));
    if ((int)rsp_n_tok != n_tokens) return -1;

    const size_t out_bytes = (size_t)n_tokens * (size_t)DS4_SHARD_N_EMBD * sizeof(float);
    if (full_recv(s->fd, out_batch, out_bytes) != 0) return -1;

    if (profile) {
        const double t_done = shard_now_sec();
        fprintf(stderr,
                "ds4_moe_shard: batch profile host=%s:%u layer=%u tokens=%d selected=%d send=%.3f ms wait_hdr=%.3f ms recv_out=%.3f ms total=%.3f ms\n",
                s->host,
                s->port,
                layer,
                n_tokens,
                n_selected,
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
    bool      is_batch;        /* false = single-token, true = batch */

    /* Single-token fields. */
    const void *xq;
    uint16_t  expert_ids[DS4_SHARD_N_EXPERT_USED];
    float     expert_weights[DS4_SHARD_N_EXPERT_USED];
    int       n_experts;
    float    *out_ptr;
    float     out[DS4_SHARD_N_EMBD];

    /* Batch fields (pointers owned by caller, valid only while has_work). */
    const void *batch_xq;          /* n_tokens × Q8K_BYTES */
    uint16_t   *batch_ids;         /* n_tokens × n_selected */
    float      *batch_wts;         /* n_tokens × n_selected */
    uint8_t    *batch_token_n;     /* n_tokens */
    int         batch_n_tokens;
    int         batch_n_selected;
    float      *batch_out;         /* n_tokens × N_EMBD (caller-allocated) */

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
    /* Persistent scratch for batch dispatch — grows only. */
    uint8_t *batch_scratch;
    size_t   batch_scratch_cap;
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

        if (w->is_batch) {
            /* Batch dispatch to remote shard. */
            w->result = ds4_shard_dispatch_layer_batch(
                w->shard, w->layer,
                w->batch_xq, w->batch_ids, w->batch_wts,
                w->batch_token_n, w->batch_n_tokens,
                w->batch_n_selected, w->batch_out);
        } else if (w->n_experts > 0) {
            w->result = ds4_shard_dispatch_layer(
                w->shard, w->layer, w->xq,
                w->expert_ids, w->expert_weights, w->n_experts,
                w->out_ptr ? w->out_ptr : w->out);
        } else {
            w->result = 0;
            memset(w->out_ptr ? w->out_ptr : w->out, 0, sizeof(w->out));
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
    free(p->batch_scratch);
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
        w->is_batch = false;
        w->xq = xq;
        w->n_experts = 0;
        w->out_ptr = (p->n_shards == 1) ? out : NULL;

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
    if (p->n_shards != 1) memset(out, 0, (size_t)DS4_SHARD_N_EMBD * sizeof(float));

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

        if (w->n_experts > 0 && p->n_shards != 1) {
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

int ds4_shard_pool_dispatch_layer_batch(
    ds4_shard_pool   *p,
    uint8_t           layer,
    const void       *xq,
    const int        *selected,
    const float      *weights,
    int               n_tokens,
    int               n_selected,
    float            *out)
{
    if (!p || !xq || !selected || !weights || !out) return -1;
    if (n_tokens <= 0 || n_selected <= 0 || n_selected > DS4_SHARD_N_EXPERT_USED) return -1;

    const int profile = shard_profile_enabled();
    const double t0 = profile ? shard_now_sec() : 0.0;
    memset(out, 0, (size_t)n_tokens * (size_t)DS4_SHARD_N_EMBD * sizeof(float));

    const size_t tok_row = (size_t)n_tokens * (size_t)n_selected;
    const size_t out_floats = (size_t)n_tokens * (size_t)DS4_SHARD_N_EMBD;

    /* Per-shard scratch: token_n + ids + wts + partial output. */
    const size_t per_shard_bytes =
        (size_t)n_tokens * sizeof(uint8_t) +
        tok_row * sizeof(uint16_t) +
        tok_row * sizeof(float) +
        out_floats * sizeof(float);
    const size_t scratch_bytes = (size_t)p->n_shards * per_shard_bytes;
    if (scratch_bytes > p->batch_scratch_cap) {
        free(p->batch_scratch);
        p->batch_scratch = malloc(scratch_bytes);
        if (!p->batch_scratch) {
            p->batch_scratch_cap = 0;
            return -1;
        }
        p->batch_scratch_cap = scratch_bytes;
    }
    uint8_t *scratch = p->batch_scratch;

    /* ---- Prepare: filter experts per shard, set up worker fields ---- */
    int n_active = 0;    /* how many shards actually have work */
    int active_si[32];   /* shard indices with work (max 32 shards) */

    for (int si = 0; si < p->n_shards; si++) {
        shard_worker *w = &p->workers[si];
        uint8_t *base = scratch + (size_t)si * per_shard_bytes;
        uint8_t  *token_n = base;
        uint16_t *ids = (uint16_t *)(base + (size_t)n_tokens * sizeof(uint8_t));
        float    *wts = (float *)((uint8_t *)ids + tok_row * sizeof(uint16_t));
        float    *partial = (float *)((uint8_t *)wts + tok_row * sizeof(float));

        int any = 0;
        for (int t = 0; t < n_tokens; t++) {
            int k = 0;
            const int *sel_row = selected + (size_t)t * (size_t)n_selected;
            const float *wt_row = weights + (size_t)t * (size_t)n_selected;
            uint16_t *id_row = ids + (size_t)t * (size_t)n_selected;
            float *w_row = wts + (size_t)t * (size_t)n_selected;
            for (int ei = 0; ei < n_selected; ei++) {
                if (ds4_shard_owns(w->shard, sel_row[ei])) {
                    id_row[k] = (uint16_t)sel_row[ei];
                    w_row[k] = wt_row[ei];
                    k++;
                }
            }
            token_n[t] = (uint8_t)k;
            for (int j = k; j < n_selected; j++) {
                id_row[j] = 0;
                w_row[j] = 0.0f;
            }
            if (k > 0) any = 1;
        }

        if (!any) continue;

        /* Set up batch work for this shard's worker thread. */
        w->layer = layer;
        w->is_batch = true;
        w->batch_xq = xq;
        w->batch_ids = ids;
        w->batch_wts = wts;
        w->batch_token_n = token_n;
        w->batch_n_tokens = n_tokens;
        w->batch_n_selected = n_selected;
        w->batch_out = (p->n_shards == 1) ? out : partial;
        active_si[n_active++] = si;
    }

    /* ---- Fire: signal all active shard workers in parallel ---- */
    for (int i = 0; i < n_active; i++) {
        shard_worker *w = &p->workers[active_si[i]];
        pthread_mutex_lock(&w->mutex);
        w->done = false;
        w->has_work = true;
        pthread_cond_signal(&w->cond);
        pthread_mutex_unlock(&w->mutex);
    }

    /* ---- Join: wait for all workers, accumulate partials ---- */
    int rc = 0;
    for (int i = 0; i < n_active; i++) {
        shard_worker *w = &p->workers[active_si[i]];
        pthread_mutex_lock(&w->mutex);
        while (!w->done)
            pthread_cond_wait(&w->cond, &w->mutex);
        pthread_mutex_unlock(&w->mutex);

        if (w->result != 0) {
            fprintf(stderr, "ds4_moe_shard: batch dispatch failed for shard %d layer %u\n",
                    active_si[i], layer);
            rc = -1;
            break;
        }

        if (p->n_shards != 1) {
            for (size_t j = 0; j < out_floats; j++)
                out[j] += w->batch_out[j];
        }
    }

    if (profile) {
        const double t_done = shard_now_sec();
        fprintf(stderr,
                "ds4_moe_shard: pool batch profile layer=%u shards=%d(%d active) tokens=%d selected=%d total=%.3f ms\n",
                layer,
                p->n_shards,
                n_active,
                n_tokens,
                n_selected,
                (t_done - t0) * 1000.0);
    }
    return rc;
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
