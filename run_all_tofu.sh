#!/bin/bash
# TOFU-Only Experiments (Reverse Order)
# 順序: idk → KL → grad_ascent → grad_diff
#
# ⚠️ 先改 configs 的 layer_search:
#    start: 12
#    end: 13
#    step: 1

set -e

echo "============================================================"
echo "TOFU EXPERIMENTS (Reverse Order)"
echo "============================================================"
echo ""

# [1/4] IDK
echo "[1/4] TOFU - IDK..."
python src/pipeline.py --config configs/config_tofu_idk.yaml --output ./outputs_tofu_4_idk --phase 0
python src/analysis/correlation_analysis.py ./outputs_tofu_4_idk
rm -rf ~/.cache/huggingface/hub/models--locuslab--phi_idk_1e-05_forget10
echo "[1/4] Done"

# [2/4] KL
echo "[2/4] TOFU - KL..."
python src/pipeline.py --config configs/config_tofu_KL.yaml --output ./outputs_tofu_3_KL --phase 0
python src/analysis/correlation_analysis.py ./outputs_tofu_3_KL
rm -rf ~/.cache/huggingface/hub/models--locuslab--phi_KL_1e-05_forget10
echo "[2/4] Done"

# [3/4] Gradient Ascent
echo "[3/4] TOFU - Gradient Ascent..."
python src/pipeline.py --config configs/config_tofu_grad_ascent.yaml --output ./outputs_tofu_2_grad_ascent --phase 0
python src/analysis/correlation_analysis.py ./outputs_tofu_2_grad_ascent
rm -rf ~/.cache/huggingface/hub/models--locuslab--phi_grad_ascent_1e-05_forget10
echo "[3/4] Done"

# [4/4] Gradient Diff
echo "[4/4] TOFU - Gradient Diff..."
python src/pipeline.py --config configs/config_tofu_grad_diff.yaml --output ./outputs_tofu_1_grad_diff --phase 0
python src/analysis/correlation_analysis.py ./outputs_tofu_1_grad_diff
echo "[4/4] Done"

# Cleanup
rm -rf ~/.cache/huggingface/hub/models--locuslab--*
rm -rf ~/.cache/huggingface/hub/models--microsoft--phi*

echo ""
echo "============================================================"
echo "COMPLETE! Check correlation_summary.md in each output folder"
echo "============================================================"