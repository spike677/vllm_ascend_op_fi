#include <iostream>
#include "flash_attention_score_template_tiling_key.h"
int main() {
    // Exact S1S2 GetTilingKey parameters, not direct bitfield guesses.
    auto key = GET_TPL_TILING_KEY(0, 3, 4, 9, 0, 3, 3, 0, 0, 2, 2, 0, 1, 0, 1, 0, 0, 0, 0, 0);
    std::cout << "{\"schema\":\"fault28.fa.sdk_key_check.v1\",\"key\":" << key
              << ",\"kernel_class\":\"FlashAttentionScoreS1s2Bn2gs1\",\"input\":\"half\",\"accumulation\":\"float\",\"bmm1_format\":\"ND\",\"l1_reuse\":true,\"impl_mode\":0,\"layout\":\"BNSD\",\"key_sparse_dispatch\":\"ANY\",\"actual_sparse_from_raw\":\"CAUSAL\"}\n";
    return key == 22619644976ULL ? 0 : 1;
}
