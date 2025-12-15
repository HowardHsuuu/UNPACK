#!/bin/bash
# Full Experiment Runner Script
# 5 Experiments: 1 Harry Potter + 4 TOFU methods

set -e  # Exit on error

echo "============================================================"
echo "MACHINE UNLEARNING FULL EXPERIMENT SUITE"
echo "5 Experiments: HP + TOFU (grad_diff, grad_ascent, KL, idk)"
echo "============================================================"
echo ""

# ============================================================
# EXPERIMENT 1: Harry Potter (Llama-2-7b)
# ============================================================
echo "[1/5] Starting Harry Potter Experiment..."
echo "Models: Llama-2-7b-chat-hf -> Llama2-7b-WhoIsHarryPotter"
echo ""

python src/pipeline.py \
    --config configs/config_hp_full.yaml \
    --output ./outputs_hp_full \
    --phase 0

echo ""
echo "[1/5] Harry Potter complete! Results: ./outputs_hp_full"
echo ""

# CLEANUP - Remove ALL HP models to free space for TOFU
echo "[CLEANUP] Removing Harry Potter models..."
rm -rf ~/.cache/huggingface/hub/models--meta-llama--Llama-2-7b-chat-hf
rm -rf ~/.cache/huggingface/hub/models--microsoft--Llama2-7b-WhoIsHarryPotter
rm -rf ~/.cache/huggingface/hub/models--muse-bench--*
echo "Done."
echo ""

# ============================================================
# EXPERIMENT 2: TOFU - Gradient Difference
# ============================================================
echo "[2/5] Starting TOFU #1 (Gradient Difference)..."
echo "Models: tofu_ft_phi-1.5 -> phi_grad_diff_1e-05_forget10"
echo ""

python src/pipeline.py \
    --config configs/config_tofu_grad_diff.yaml \
    --output ./outputs_tofu_1_grad_diff \
    --phase 0

echo ""
echo "[2/5] TOFU grad_diff complete! Results: ./outputs_tofu_1_grad_diff"
echo ""

# CLEANUP - Only remove unlearned model, KEEP base model for next runs
echo "[CLEANUP] Removing grad_diff unlearned model (keeping base)..."
rm -rf ~/.cache/huggingface/hub/models--locuslab--phi_grad_diff_1e-05_forget10
echo "Done."
echo ""

# ============================================================
# EXPERIMENT 3: TOFU - Gradient Ascent
# ============================================================
echo "[3/5] Starting TOFU #2 (Gradient Ascent)..."
echo "Models: tofu_ft_phi-1.5 (cached) -> phi_grad_ascent_1e-05_forget10"
echo ""

python src/pipeline.py \
    --config configs/config_tofu_grad_ascent.yaml \
    --output ./outputs_tofu_2_grad_ascent \
    --phase 0

echo ""
echo "[3/5] TOFU grad_ascent complete! Results: ./outputs_tofu_2_grad_ascent"
echo ""

# CLEANUP
echo "[CLEANUP] Removing grad_ascent unlearned model (keeping base)..."
rm -rf ~/.cache/huggingface/hub/models--locuslab--phi_grad_ascent_1e-05_forget10
echo "Done."
echo ""

# ============================================================
# EXPERIMENT 4: TOFU - KL Divergence
# ============================================================
echo "[4/5] Starting TOFU #3 (KL Divergence)..."
echo "Models: tofu_ft_phi-1.5 (cached) -> phi_KL_1e-05_forget10"
echo ""

python src/pipeline.py \
    --config configs/config_tofu_KL.yaml \
    --output ./outputs_tofu_3_KL \
    --phase 0

echo ""
echo "[4/5] TOFU KL complete! Results: ./outputs_tofu_3_KL"
echo ""

# CLEANUP
echo "[CLEANUP] Removing KL unlearned model (keeping base)..."
rm -rf ~/.cache/huggingface/hub/models--locuslab--phi_KL_1e-05_forget10
echo "Done."
echo ""

# ============================================================
# EXPERIMENT 5: TOFU - IDK (I Don't Know)
# ============================================================
echo "[5/5] Starting TOFU #4 (IDK)..."
echo "Models: tofu_ft_phi-1.5 (cached) -> phi_idk_1e-05_forget10"
echo ""

python src/pipeline.py \
    --config configs/config_tofu_idk.yaml \
    --output ./outputs_tofu_4_idk \
    --phase 0

echo ""
echo "[5/5] TOFU idk complete! Results: ./outputs_tofu_4_idk"
echo ""

# FINAL CLEANUP - Now remove everything
echo "[CLEANUP] Removing all remaining models..."
rm -rf ~/.cache/huggingface/hub/models--locuslab--*
rm -rf ~/.cache/huggingface/hub/models--microsoft--phi*
echo "Done."
echo ""

# ============================================================
# SUMMARY
# ============================================================
echo "============================================================"
echo "ALL 5 EXPERIMENTS COMPLETE!"
echo "============================================================"
echo ""
echo "Results:"
echo "  1. Harry Potter:        ./outputs_hp_full"
echo "  2. TOFU grad_diff:      ./outputs_tofu_1_grad_diff"
echo "  3. TOFU grad_ascent:    ./outputs_tofu_2_grad_ascent"
echo "  4. TOFU KL:             ./outputs_tofu_3_KL"
echo "  5. TOFU idk:            ./outputs_tofu_4_idk"
echo ""
echo "Key comparison files to examine:"
echo "  - attack_comparison.csv"
echo "  - 2x2_summary.csv"
echo "  - logit_lens_layer_summary.csv"
echo ""