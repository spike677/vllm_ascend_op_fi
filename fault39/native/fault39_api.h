#pragma once
#include <stdint.h>
#include <stddef.h>
#ifdef __cplusplus
extern "C" {
#endif
enum Fault39ErrorV1 {
 F39_OK=0, F39_INVALID_CONFIG=-1, F39_ABI_MISMATCH=-2, F39_BUSY=-3,
 F39_OWNER_MISMATCH=-4, F39_GENERATION_MISMATCH=-5, F39_AMBIGUOUS_TARGET=-6,
 F39_TARGET_NOT_FOUND=-7, F39_UNSUPPORTED_RUNTIME=-8, F39_RECOVERY_FAILED=-9,
 F39_DRAIN_TIMEOUT=-10, F39_NO_OVERFLOW_SOLUTION=-11
};
enum Fault39ModeV1 { F39_DISCOVER=1, F39_INJECT=2, F39_LOSSLESS=3, F39_OBSERVE=4 };
enum Fault39PatternV1 { F39_HALF=0, F39_ALTERNATING=1, F39_BLOCK256=2, F39_BLOCK512=3, F39_BLOCK1024=4 };
typedef struct Fault39ConfigV1 {
 uint32_t abi_version, struct_size, mode, pattern;
 int32_t exponent, rank, device;
 uint32_t max_hits;
 uint64_t max_bytes;
 char owner[64];
 char target_signature[256];
 uint32_t input_mask, reserved;
} Fault39ConfigV1;
uint32_t fault39_abi_version_v1(void);
/* Additive diagnostic: immutable identity of this exact toolkit/header build. */
const char* fault39_build_profile_v1(void);
int fault39_configure_v1(const Fault39ConfigV1*, uint64_t* generation);
/* Additive exact cache registration identity; v1 unique-signature rules remain.
 * Requires same-thread live serving inputs, two real observations of this cache
 * epoch, and max_hits=1. Address is a lookup key, never a pointer to dereference.
 */
int fault39_configure_cache_v2(const Fault39ConfigV1*, uint64_t cache_address,
                             uint64_t cache_epoch, uint64_t* generation);
int fault39_disable_v1(const char* owner, uint64_t generation, uint32_t timeout_ms);
int fault39_clean_v1(const char* owner, uint64_t generation, uint32_t timeout_ms);
/* Returns required bytes INCLUDING NUL; insufficient buffers are never partially written. */
int fault39_snapshot_v1(char* out, size_t capacity);
/* Thread-scoped actual scheduler request IDs; clear on every exit, never inherited. */
int fault39_set_context_v1(const char* owner, const char* request_ids_json);
/* Optional serving guard, set once within the same owner-bound TLS context.
 * Exactly 13 uint64 values: self address/bytes, mat2 address/bytes, M/K/N,
 * self/mat2 storage offsets, self strides 0/1, mat2 strides 0/1.
 * Addresses are storage bases; sizes are bytes; offsets/strides are elements.
 * A rejected binding by the current owner remains guarded (fail closed).
 */
int fault39_set_serving_inputs_v1(const char* owner, const uint64_t* values, size_t count);
void fault39_clear_context_v1(void);
int fault39_bind_worker_rank_v1(int32_t worker_rank);
#ifdef __cplusplus
}
static_assert(sizeof(Fault39ConfigV1)==368,"Fault39 v1 configuration ABI");
#endif
