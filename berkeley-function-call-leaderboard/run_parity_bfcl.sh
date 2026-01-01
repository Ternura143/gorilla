#!/bin/bash
# BFCL Parity Test Script - Run on Codespaces
# Usage: ./run_parity_bfcl.sh <model> <runs>
# Example: ./run_parity_bfcl.sh codex-gpt-5-mini 5

set -e

MODEL=${1:-codex-gpt-5-mini}
RUNS=${2:-5}
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_FILE="parity_results_${MODEL}_${TIMESTAMP}.txt"

echo "=========================================="
echo "BFCL Parity Test"
echo "Model: $MODEL"
echo "Runs: $RUNS"
echo "Results: $RESULTS_FILE"
echo "=========================================="

# Check if OPENAI_API_KEY is set
if [ -z "$OPENAI_API_KEY" ]; then
    echo "Error: OPENAI_API_KEY is not set"
    echo "Run: export OPENAI_API_KEY='your-key'"
    exit 1
fi

# Check if test_case_ids_to_generate.json exists
if [ ! -f "test_case_ids_to_generate.json" ]; then
    echo "Error: test_case_ids_to_generate.json not found"
    echo "Please create it with the parity sample IDs"
    exit 1
fi

echo "" > "$RESULTS_FILE"
echo "Parity Test Results - $MODEL" >> "$RESULTS_FILE"
echo "Date: $(date)" >> "$RESULTS_FILE"
echo "========================================" >> "$RESULTS_FILE"

for i in $(seq 1 $RUNS); do
    echo ""
    echo ">>> Run $i of $RUNS"
    echo ">>> $(date)"
    
    # Clean previous results
    rm -rf "result/$MODEL" "score/$MODEL"
    
    # Generate
    echo "Generating responses..."
    python -m bfcl_eval generate --model "$MODEL" --run-ids -o
    
    # Evaluate
    echo "Evaluating..."
    python -m bfcl_eval evaluate --model "$MODEL" --partial-eval
    
    # Extract results from score files
    echo "Extracting results..."
    
    # Count results
    total=0
    correct=0
    
    for score_file in score/$MODEL/*/score_*.json; do
        if [ -f "$score_file" ]; then
            file_correct=$(jq '[.[] | select(.valid == true)] | length' "$score_file" 2>/dev/null || echo 0)
            file_total=$(jq '. | length' "$score_file" 2>/dev/null || echo 0)
            correct=$((correct + file_correct))
            total=$((total + file_total))
        fi
    done
    
    if [ $total -gt 0 ]; then
        accuracy=$(echo "scale=2; $correct * 100 / $total" | bc)
        echo "Run $i: $accuracy% ($correct/$total)" | tee -a "$RESULTS_FILE"
    else
        echo "Run $i: No results found" | tee -a "$RESULTS_FILE"
    fi
    
    # Save detailed score
    cp -r "score/$MODEL" "score/${MODEL}_run${i}_${TIMESTAMP}" 2>/dev/null || true
    
    echo ">>> Run $i completed"
done

echo ""
echo "=========================================="
echo "All runs completed!"
echo "Results saved to: $RESULTS_FILE"
echo "=========================================="
cat "$RESULTS_FILE"

