#ifndef DS4_MOE_SHARD_H
#define DS4_MOE_SHARD_H

/*
 * MoE Expert Sharding — client-side shard dispatch.
 *
 * Offloads routed expert computation (gate/up/SwiGLU/down) to CPU-only remote
 * machines over persistent TCP connections.  The coordinator keeps routing,
 * attention, shared expert, and norms local.
 *
 * Wire protocol:  binary over raw TCP, no HTTP framing.
 * Threading:      one persistent worker pthread per shard, barrier-synchronized.
 */

#include <stdbool.h>
#include <stdint.h>

/* ---- Wire protocol constants ------------------------------------------- */

#define DS4_SHARD_MAGIC        0x44533445u  /* "DS4E" little-endian */
#define DS4_SHARD_CMD_PING     0x00
#define DS4_SHARD_CMD_EXPERT   0x01
#define DS4_SHARD_CMD_EXPERT_BATCH 0x02
#define DS4_SHARD_CMD_SHUTDOWN 0xFF

#define DS4_SHARD_STATUS_OK    0x00
#define DS4_SHARD_STATUS_ERR   0x01

/* DS4 fixed model shape — duplicated here so the header is self-contained. */
#define DS4_SHARD_N_EMBD       4096
#define DS4_SHARD_N_EXPERT     256
#define DS4_SHARD_N_EXPERT_USED 6
#define DS4_SHARD_N_FF_EXP     2048
#define DS4_SHARD_N_LAYER      43

/* Q8_K block: 292 bytes per 256 elements.  Activation is 4096 elements →
 * 16 blocks → 4672 bytes. */
#define DS4_SHARD_Q8K_BLOCKS   (DS4_SHARD_N_EMBD / 256)
#define DS4_SHARD_Q8K_BYTES    (DS4_SHARD_Q8K_BLOCKS * 292)

/* ---- Request/Response wire layout -------------------------------------- */

/*
 * Request header (coordinator → shard):
 *   [magic     u32]  DS4_SHARD_MAGIC
 *   [cmd       u8 ]  DS4_SHARD_CMD_EXPERT
 *   [layer     u8 ]  0..42
 *   [n_expert  u8 ]  1..6
 *   [flags     u8 ]  bit 0: activation is Q8_K (else f32)
 *
 * Request body:
 *   [expert_ids     n_expert × u16]
 *   [expert_weights n_expert × f32]
 *   [activation     DS4_SHARD_Q8K_BYTES or DS4_SHARD_N_EMBD*4 bytes]
 *
 * Response:
 *   [magic     u32]  DS4_SHARD_MAGIC
 *   [status    u8 ]  DS4_SHARD_STATUS_OK or _ERR
 *   [layer     u8 ]
 *   [pad       u16]
 *   [output    DS4_SHARD_N_EMBD × f32]  (weighted partial sum)
 */

#define DS4_SHARD_REQ_HDR_SIZE  8   /* magic(4) + cmd(1) + layer(1) + n_expert(1) + flags(1) */
#define DS4_SHARD_RSP_HDR_SIZE  8   /* magic(4) + status(1) + layer(1) + pad(2) */
#define DS4_SHARD_FLAG_Q8K      0x01
/* Batch request/response headers.
 * req: magic(4) + cmd(1) + layer(1) + flags(1) + pad(1) + n_tokens(2) + n_selected(2)
 * rsp: magic(4) + status(1) + layer(1) + pad(2) + n_tokens(2) + pad(2) */
#define DS4_SHARD_BATCH_REQ_HDR_SIZE 12
#define DS4_SHARD_BATCH_RSP_HDR_SIZE 12

/* ---- Single shard connection ------------------------------------------- */

typedef struct ds4_shard ds4_shard;

typedef struct {
    const char *host;
    uint16_t    port;
    uint16_t    expert_start;  /* first expert ID owned (inclusive) */
    uint16_t    expert_end;    /* last expert ID owned (exclusive) */
} ds4_shard_config;

int  ds4_shard_connect(ds4_shard **out, const ds4_shard_config *cfg);
void ds4_shard_close(ds4_shard *s);
bool ds4_shard_ping(ds4_shard *s);
bool ds4_shard_owns(const ds4_shard *s, int expert_id);

/* Dispatch one layer's experts to this shard.  Only experts this shard owns
 * are included.  xq is a Q8_K-quantized activation (DS4_SHARD_Q8K_BYTES).
 * out[DS4_SHARD_N_EMBD] receives the shard's weighted partial sum. */
int ds4_shard_dispatch_layer(
    ds4_shard        *s,
    uint8_t           layer,
    const void       *xq,             /* Q8_K activation bytes */
    const uint16_t   *expert_ids,     /* subset owned by this shard */
    const float      *expert_weights,
    int               n_experts,
    float            *out);

/* ---- Shard pool: multi-shard parallel fan-out -------------------------- */

typedef struct ds4_shard_pool ds4_shard_pool;

int  ds4_shard_pool_create(ds4_shard_pool **out,
                           const ds4_shard_config *cfgs, int n_shards);
void ds4_shard_pool_close(ds4_shard_pool *p);

/* Route-and-dispatch for one layer.  Partitions selected experts by shard
 * ownership, dispatches to all relevant shards in parallel, and sums the
 * partial results into out[DS4_SHARD_N_EMBD]. */
int ds4_shard_pool_dispatch_layer(
    ds4_shard_pool   *p,
    uint8_t           layer,
    const void       *xq,
    const int        *selected,       /* DS4_SHARD_N_EXPERT_USED expert IDs */
    const float      *weights,        /* DS4_SHARD_N_EXPERT_USED weights */
    int               n_selected,
    float            *out);

/* Layer-batch dispatch. selected/weights are laid out as n_tokens rows, each
 * with n_selected entries. xq is n_tokens consecutive Q8_K activations.
 * out is n_tokens consecutive DS4_SHARD_N_EMBD float rows. */
int ds4_shard_pool_dispatch_layer_batch(
    ds4_shard_pool   *p,
    uint8_t           layer,
    const void       *xq,
    const int        *selected,
    const float      *weights,
    int               n_tokens,
    int               n_selected,
    float            *out);

/* Parse "host1:port:start-end,host2:port:start-end,..." into an array of
 * configs.  Caller frees the returned array.  Returns count, or -1 on error. */
int ds4_shard_config_parse(const char *spec, ds4_shard_config **out);

#endif /* DS4_MOE_SHARD_H */
