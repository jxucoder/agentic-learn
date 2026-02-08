#!/bin/bash
# Entry point for the agentic-learn MLE-bench agent.
# Called by MLE-bench via: bash /home/agent/start.sh

set -e

AGENT_DIR="${AGENT_DIR:-/home/agent}"
SUBMISSION_DIR="${SUBMISSION_DIR:-/home/submission}"
LOGS_DIR="${LOGS_DIR:-/home/logs}"
CODE_DIR="${CODE_DIR:-/home/code}"

echo "=== agentic-learn MLE-bench Agent ==="
echo "Agent dir: ${AGENT_DIR}"
echo "Submission dir: ${SUBMISSION_DIR}"
echo "Code dir: ${CODE_DIR}"
echo "Competition: ${COMPETITION_ID:-unknown}"

# Create directories
mkdir -p "${SUBMISSION_DIR}" "${LOGS_DIR}" "${CODE_DIR}"

# Run the agent
/opt/conda/bin/conda run -n mleb python "${AGENT_DIR}/src/agent.py" \
    --data-dir /home/data \
    --submission-dir "${SUBMISSION_DIR}" \
    --code-dir "${CODE_DIR}" \
    --logs-dir "${LOGS_DIR}" \
    --time-limit "${TIME_LIMIT_SECS:-86400}" \
    2>&1 | tee "${LOGS_DIR}/agent.log"

echo "=== Agent finished ==="
