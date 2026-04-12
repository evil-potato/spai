#!/bin/bash

# ==========================================
# 定义模型权重列表
# ==========================================
MODELS=(
    # "./output/train/finetune/spai/ckpt_epoch_13.pth"
    # "./output/train/finetune/spai/ckpt_epoch_15.pth"
    "./output/train_cuda_3/finetune/spai/ckpt_epoch_6.pth"
    "./output/train_cuda_3/finetune/spai/ckpt_epoch_7.pth"
    # "./weights/spai.pth"
)

# ==========================================
# 定义测试CSV列表
# ==========================================
TEST_CSVS=(
    "./data/fake_dalle2.csv"
    "./data/fake_dalle3.csv"
    "./data/fake_firefly.csv"
    "./data/fake_flux.csv"
    "./data/fake_gigagan.csv"
    "./data/fake_glide.csv"
    "./data/fake_mjv5.csv"
    "./data/fake_mjv61.csv"
    "./data/fake_sd2.csv"
    "./data/fake_sd3.csv"
    "./data/fake_sd13.csv"
    "./data/fake_sd14.csv"
    "./data/fake_sdxl.csv"
    "./data/real_coco.csv"
    "./data/real_fodb.csv"
    "./data/real_imagenet.csv"
    "./data/real_openimages.csv"
    "./data/real_raise.csv"

)

# ==========================================
# 公共参数
# ==========================================
CUDA_VISIBLE_DEVICES=3
CFG="./configs/spai.yaml"
BATCH_SIZE=8
BASE_OUTPUT="./output/test"
TAG="spai"

# 统计变量
TOTAL=$((${#MODELS[@]} * ${#TEST_CSVS[@]}))
COUNT=0
FAILED=0

echo "=========================================="
echo "批量测试启动"
echo "模型数量: ${#MODELS[@]}"
echo "数据集数量: ${#TEST_CSVS[@]}"
echo "总任务数: ${TOTAL}"
echo "开始时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================="

# ==========================================
# 双重循环：遍历模型 × 遍历数据集
# ==========================================
for MODEL_PATH in "${MODELS[@]}"; do
    # 提取模型名（如 ckpt_epoch_7）
    MODEL_NAME=$(basename "${MODEL_PATH}" .pth)

    echo ""
    echo "****************************************"
    echo "当前模型: ${MODEL_NAME}"
    echo "****************************************"

    for CSV_PATH in "${TEST_CSVS[@]}"; do
        COUNT=$((COUNT + 1))
        CSV_NAME=$(basename "${CSV_PATH}" .csv)

        # 输出目录按 模型/数据集 分层组织
        OUTPUT_DIR="${BASE_OUTPUT}/${MODEL_NAME}"

        echo ""
        echo "  [${COUNT}/${TOTAL}] 模型: ${MODEL_NAME} | 数据集: ${CSV_NAME}"
        echo "  输出目录: ${OUTPUT_DIR}"
        echo "  开始时间: $(date '+%Y-%m-%d %H:%M:%S')"

        CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} python -m spai infer \
            --cfg "${CFG}" \
            --batch-size ${BATCH_SIZE} \
            --model "${MODEL_PATH}" \
            --output "${OUTPUT_DIR}" \
            --tag "${TAG}" \
            --opt "MODEL.PATCH_VIT.MINIMUM_PATCHES" "4" \
            --opt "DATA.NUM_WORKERS" "8" \
            --opt "MODEL.FEATURE_EXTRACTION_BATCH" "64" \
            --opt "DATA.TEST_PREFETCH_FACTOR" "1" \
            --input "${CSV_PATH}"

        if [ $? -eq 0 ]; then
            echo "  ✓ 完成 [${COUNT}/${TOTAL}] - $(date '+%Y-%m-%d %H:%M:%S')"
        else
            FAILED=$((FAILED + 1))
            echo "  ✗ 失败 [${COUNT}/${TOTAL}] - $(date '+%Y-%m-%d %H:%M:%S')"
            # 失败后停止所有测试，取消下一行注释
            # exit 1
        fi

    done

    python calc_final_metrics.py \
    --scores-dir "${OUTPUT_DIR}" \
    --output-csv "${OUTPUT_DIR}/final_results.csv" \
    --tag-column "${TAG}"

    echo ""
    echo "  ✓ 模型 [${MODEL_NAME}] 全部数据集测试完毕"

done

# ==========================================
# 汇总报告
# ==========================================
echo ""
echo "=========================================="
echo "所有测试执行完毕"
echo "总任务: ${TOTAL}"
echo "成功:   $((TOTAL - FAILED))"
echo "失败:   ${FAILED}"
echo "结束时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================="