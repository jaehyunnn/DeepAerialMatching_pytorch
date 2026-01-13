#!/bin/bash
# =============================================================================
# Deep Aerial Matching - Session Evaluation Script
# =============================================================================
#
# Evaluates all checkpoints in a training session folder
#
# Usage:
#   ./scripts/eval_session.sh <session_folder> [OPTIONS]
#   ./scripts/eval_session.sh checkpoints/20250112_143022
#   ./scripts/eval_session.sh checkpoints/20250112_143022 --batch-size 32
#
# =============================================================================

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

# Show help if requested or no arguments
if [[ $# -eq 0 || "$1" == "-h" || "$1" == "--help" ]]; then
    echo -e "${GREEN}Deep Aerial Matching - Session Evaluation${NC}"
    echo ""
    echo "Evaluates all checkpoints (.pt files) in a training session folder"
    echo ""
    echo -e "${YELLOW}Usage:${NC}"
    echo "  ./scripts/eval_session.sh <session_folder> [OPTIONS]"
    echo ""
    echo -e "${YELLOW}Examples:${NC}"
    echo "  ./scripts/eval_session.sh checkpoints/20250112_143022"
    echo "  ./scripts/eval_session.sh checkpoints/20250112_143022 --batch-size 32"
    echo "  ./scripts/eval_session.sh checkpoints/20250112_143022 --num-workers 8"
    echo ""
    echo -e "${YELLOW}Additional options (passed to eval.sh):${NC}"
    echo "  --backbone MODEL        resnet101|resnext101|se_resnext101|densenet169|vit-l/16"
    echo "  --version VERSION       v1|v2"
    echo "  --dataset-path PATH     Path to evaluation dataset"
    echo "  --batch-size N          Evaluation batch size"
    echo "  --num-workers N         Data loading workers"
    echo ""

fi

# Get session folder from first argument
SESSION_FOLDER="$1"
shift

# Check if session folder exists
if [ ! -d "$SESSION_FOLDER" ]; then
    echo -e "${RED}Error: Session folder not found: $SESSION_FOLDER${NC}"
    echo ""
    echo -e "${YELLOW}Available sessions:${NC}"
    ls -d checkpoints/*/ 2>/dev/null | sed 's/checkpoints\//  /' | sed 's/\///' || echo "  (no sessions found)"
    exit 1
fi

# Find all .pt files in the session folder (sort numerically)
CHECKPOINTS=($(find "$SESSION_FOLDER" -name "*.pt" -type f | sort -V))

if [ ${#CHECKPOINTS[@]} -eq 0 ]; then
    echo -e "${RED}Error: No checkpoint files (.pt) found in $SESSION_FOLDER${NC}"
    exit 1
fi

# Extract additional arguments to pass to eval.sh
EVAL_ARGS="$@"

# Print header
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  Session Evaluation${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo -e "${BLUE}Session folder:${NC} $SESSION_FOLDER"
echo -e "${BLUE}Checkpoints found:${NC} ${#CHECKPOINTS[@]}"
echo ""

# List all checkpoints
echo -e "${CYAN}Checkpoints to evaluate:${NC}"
for checkpoint in "${CHECKPOINTS[@]}"; do
    filename=$(basename "$checkpoint")
    echo "  - $filename"
done
echo ""

# Ask for confirmation
read -p "$(echo -e ${YELLOW}Continue with evaluation? [Y/n]:${NC} )" -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]] && [[ -n $REPLY ]]; then
    echo -e "${RED}Cancelled.${NC}"
    exit 0
fi

# Create results directory
RESULTS_DIR="$SESSION_FOLDER/eval_results"
mkdir -p "$RESULTS_DIR"

# Evaluation loop
echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  Starting Evaluation${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

TOTAL=${#CHECKPOINTS[@]}
CURRENT=0
FAILED=0

for checkpoint in "${CHECKPOINTS[@]}"; do
    CURRENT=$((CURRENT + 1))
    filename=$(basename "$checkpoint")

    echo -e "${CYAN}[$CURRENT/$TOTAL] Evaluating: $filename${NC}"
    echo ""

    # Run eval.sh with the checkpoint
    if ./scripts/eval.sh --model "$checkpoint" $EVAL_ARGS 2>&1 | tee "$RESULTS_DIR/${filename%.pt}.log"; then
        echo -e "${GREEN}✓ Success${NC}"
    else
        echo -e "${RED}✗ Failed${NC}"
        FAILED=$((FAILED + 1))
    fi

    echo ""
    echo -e "${BLUE}----------------------------------------${NC}"
    echo ""
done

# Summary
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  Evaluation Complete${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo -e "${BLUE}Total checkpoints:${NC} $TOTAL"
echo -e "${GREEN}Successful:${NC} $((TOTAL - FAILED))"
if [ $FAILED -gt 0 ]; then
    echo -e "${RED}Failed:${NC} $FAILED"
fi
echo ""
echo -e "${BLUE}Results saved to:${NC} $RESULTS_DIR"
echo ""

# Show log files
echo -e "${CYAN}Log files:${NC}"
ls -1 "$RESULTS_DIR"/*.log 2>/dev/null | sed 's/^/  /' || echo "  (no logs found)"
echo ""

exit 0
