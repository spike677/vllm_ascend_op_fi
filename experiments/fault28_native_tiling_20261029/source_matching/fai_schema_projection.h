/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma once
#include "register/tilingdata_base.h"
namespace optiling{
    constexpr int32_t MAX_CORE_NUM_FD = 26;

    BEGIN_TILING_DATA_DEF(coreNode)
    TILING_DATA_FIELD_DEF_ARR(int32_t, MAX_CORE_NUM_FD, startBIdx)
    TILING_DATA_FIELD_DEF_ARR(int32_t, MAX_CORE_NUM_FD, startN1Idx)
    TILING_DATA_FIELD_DEF_ARR(int32_t, MAX_CORE_NUM_FD, startS1Idx)
    TILING_DATA_FIELD_DEF_ARR(int32_t, MAX_CORE_NUM_FD, startS2Idx)
    TILING_DATA_FIELD_DEF_ARR(int32_t, MAX_CORE_NUM_FD, endBIdx)
    TILING_DATA_FIELD_DEF_ARR(int32_t, MAX_CORE_NUM_FD, endN1Idx)
    TILING_DATA_FIELD_DEF_ARR(int32_t, MAX_CORE_NUM_FD, endS1Idx)
    TILING_DATA_FIELD_DEF_ARR(int32_t, MAX_CORE_NUM_FD, endS2Idx)
    TILING_DATA_FIELD_DEF_ARR(int64_t, MAX_CORE_NUM_FD, firstSplitKVTaskLseOffset)
    TILING_DATA_FIELD_DEF_ARR(int64_t, MAX_CORE_NUM_FD, firstSplitKVTaskOOffset)
    END_TILING_DATA_DEF
    REGISTER_TILING_DATA_CLASS(coreNodeOp, coreNode)

    BEGIN_TILING_DATA_DEF(splitNode)
    TILING_DATA_FIELD_DEF_ARR(int32_t, MAX_CORE_NUM_FD, batchIdx)
    TILING_DATA_FIELD_DEF_ARR(int32_t, MAX_CORE_NUM_FD, headStartIdx)
    TILING_DATA_FIELD_DEF_ARR(int32_t, MAX_CORE_NUM_FD, headEndIdx)
    TILING_DATA_FIELD_DEF_ARR(int32_t, MAX_CORE_NUM_FD, qStartIdx)
    TILING_DATA_FIELD_DEF_ARR(int32_t, MAX_CORE_NUM_FD, qEndIdx)
    TILING_DATA_FIELD_DEF_ARR(int32_t, MAX_CORE_NUM_FD, splitNum)
    TILING_DATA_FIELD_DEF_ARR(int64_t, MAX_CORE_NUM_FD, lseTaskOffset)
    TILING_DATA_FIELD_DEF_ARR(int64_t, MAX_CORE_NUM_FD, oTaskOffset)
    END_TILING_DATA_DEF
    REGISTER_TILING_DATA_CLASS(splitNodeOp, splitNode)

    BEGIN_TILING_DATA_DEF(FAInferTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, numHeads)
    TILING_DATA_FIELD_DEF(uint32_t, embeddingSize)
    TILING_DATA_FIELD_DEF(uint32_t, embeddingSizeV)
    TILING_DATA_FIELD_DEF(uint32_t, numBlocks)
    TILING_DATA_FIELD_DEF(uint32_t, blockSize)
    TILING_DATA_FIELD_DEF(uint32_t, maxQSeqlen)
    TILING_DATA_FIELD_DEF(uint32_t, maxKvSeqlen)
    TILING_DATA_FIELD_DEF(uint32_t, kvHeads)
    TILING_DATA_FIELD_DEF(uint32_t, batch)
    TILING_DATA_FIELD_DEF(uint32_t, maxNumBlocksPerBatch)
    TILING_DATA_FIELD_DEF(uint32_t, firstBatchTaskNum)
    TILING_DATA_FIELD_DEF(uint32_t, totalTaskNum)
    TILING_DATA_FIELD_DEF(uint32_t, maskType)
    TILING_DATA_FIELD_DEF(uint64_t, mm1OutSize)
    TILING_DATA_FIELD_DEF(uint64_t, smOnlineOutSize)
    TILING_DATA_FIELD_DEF(uint64_t, mm2OutSize)
    TILING_DATA_FIELD_DEF(uint64_t, UpdateSize)
    TILING_DATA_FIELD_DEF(uint64_t, workSpaceSize)
    TILING_DATA_FIELD_DEF(float, scaleValue)
    TILING_DATA_FIELD_DEF(uint64_t, pseQ)
    TILING_DATA_FIELD_DEF(uint64_t, pseKv)
    TILING_DATA_FIELD_DEF(uint32_t, padding3)
    TILING_DATA_FIELD_DEF(int64_t, preToken)
    TILING_DATA_FIELD_DEF(int64_t, nextToken)
    TILING_DATA_FIELD_DEF(uint32_t, sparseMode)
    TILING_DATA_FIELD_DEF(uint64_t, splitLseTotalSize)
    TILING_DATA_FIELD_DEF(uint64_t, splitOTotalSize)
    TILING_DATA_FIELD_DEF(uint32_t, totalSplitNodeNum)
    TILING_DATA_FIELD_DEF(uint32_t, needCoreNum)
    TILING_DATA_FIELD_DEF(uint32_t, mainLoopTaskNum)
    TILING_DATA_FIELD_DEF(uint32_t, tailLoopTaskNum)
    TILING_DATA_FIELD_DEF(uint32_t, tailStartBatch)
    TILING_DATA_FIELD_DEF(uint32_t, tailStartN2)
    TILING_DATA_FIELD_DEF(uint32_t, tailKvNBlockTile)
    TILING_DATA_FIELD_DEF_STRUCT(coreNode, coreInfo)
    TILING_DATA_FIELD_DEF_STRUCT(splitNode, splitInfo)
    END_TILING_DATA_DEF

}
