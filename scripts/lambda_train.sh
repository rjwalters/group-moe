#!/bin/bash
# High-level scripted workflow: launch Lambda instance, sync repo, run training, sync results back, terminate.
#
# Usage:
#   ./scripts/lambda_train.sh <instance-type> -- <train_qm9.py args>
#
# Examples:
#   ./scripts/lambda_train.sh gpu_1x_a100_sxm4 -- --run-name schnet_baseline --epochs 500
#   ./scripts/lambda_train.sh gpu_1x_a10 -- --run-name schnet_quick --epochs 50
#
# Environment overrides:
#   LAMBDA_REGION         Region (default: us-west-1)
#   LAMBDA_NO_TERMINATE   Set to 1 to skip auto-terminate on exit (for debugging)
#   LAMBDA_DRY_RUN        Set to 1 to print the launch payload and exit
#
# Safety:
#   - Trap on EXIT terminates the instance (unless LAMBDA_NO_TERMINATE=1)
#   - Confirms before launching (cost preview)
#   - Captures instance ID to a state file in case the script is killed
#     mid-flight; the trap reads it back to terminate

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
LAMBDA_CMD="$SCRIPT_DIR/lambda.sh"
STATE_DIR="$REPO_DIR/.lambda-state"
mkdir -p "$STATE_DIR"
STATE_FILE="$STATE_DIR/current-instance"

# --- argument parsing ---
if [ $# -lt 1 ]; then
    echo "Usage: $0 <instance-type> -- <train_qm9.py args>" >&2
    echo "Run './scripts/lambda.sh types' to see what has capacity right now." >&2
    exit 1
fi
INSTANCE_TYPE="$1"
shift
if [ "${1:-}" = "--" ]; then shift; fi
TRAIN_ARGS=("$@")
if [ ${#TRAIN_ARGS[@]} -eq 0 ]; then
    TRAIN_ARGS=(--run-name schnet_baseline --epochs 500)
    echo "[lambda_train] no train args given; using defaults: ${TRAIN_ARGS[*]}" >&2
fi

# Extract --run-name for the rsync-back step. Default mirrors train_qm9.py.
RUN_NAME=schnet_baseline
for ((i=0; i<${#TRAIN_ARGS[@]}; i++)); do
    if [ "${TRAIN_ARGS[$i]}" = "--run-name" ] && [ $((i+1)) -lt ${#TRAIN_ARGS[@]} ]; then
        RUN_NAME="${TRAIN_ARGS[$((i+1))]}"
    fi
done

REGION="${LAMBDA_REGION:-us-west-1}"

# --- cost preview ---
PRICE=$(curl -s -u "$(grep LAMBDA_API_KEY "$REPO_DIR/.env" | cut -d= -f2):" \
    "https://cloud.lambda.ai/api/v1/instance-types" \
    | jq -r ".data[\"$INSTANCE_TYPE\"].instance_type.price_cents_per_hour // empty")
if [ -z "$PRICE" ]; then
    echo "[lambda_train] WARNING: couldn't look up price for $INSTANCE_TYPE" >&2
    PRICE=200
fi
PRICE_PER_HR=$(awk "BEGIN { printf \"%.2f\", $PRICE/100 }")

cat <<EOF
================================================================
  Lambda training launch
================================================================
  Instance type:  $INSTANCE_TYPE
  Region:         $REGION
  Filesystem:     group-moe-data
  Train command:  scripts/train_qm9.py ${TRAIN_ARGS[*]}
  Run name:       $RUN_NAME
  Price:          \$$PRICE_PER_HR/hr
  Auto-terminate: ${LAMBDA_NO_TERMINATE:+disabled (LAMBDA_NO_TERMINATE=1)}${LAMBDA_NO_TERMINATE:-yes}
================================================================
EOF

if [ "${LAMBDA_DRY_RUN:-}" = "1" ]; then
    echo "[lambda_train] LAMBDA_DRY_RUN=1, exiting before launch."
    exit 0
fi

if [ "${LAMBDA_YES:-}" != "1" ]; then
    read -r -p "Proceed with launch? [y/N] " confirm
    if [ "$confirm" != "y" ] && [ "$confirm" != "Y" ]; then
        echo "Aborted." >&2
        exit 1
    fi
else
    echo "[lambda_train] LAMBDA_YES=1, skipping confirmation."
fi

# --- termination trap ---
INSTANCE_ID=""
cleanup() {
    local code=$?
    set +e
    # Read instance ID from state file in case INSTANCE_ID local var is unset
    if [ -z "$INSTANCE_ID" ] && [ -f "$STATE_FILE" ]; then
        INSTANCE_ID=$(cat "$STATE_FILE")
    fi
    if [ -z "$INSTANCE_ID" ]; then
        # Nothing to clean up
        exit $code
    fi
    if [ -n "${LAMBDA_NO_TERMINATE:-}" ]; then
        echo "[cleanup] LAMBDA_NO_TERMINATE=1; instance $INSTANCE_ID left running." >&2
        echo "[cleanup] Don't forget: ./scripts/lambda.sh terminate $INSTANCE_ID" >&2
        exit $code
    fi
    echo "[cleanup] terminating $INSTANCE_ID (orchestrator exit code $code)..." >&2
    TERM_RESPONSE=$("$LAMBDA_CMD" terminate "$INSTANCE_ID" 2>&1)
    echo "[cleanup] terminate response: $TERM_RESPONSE" >&2
    # Verify the instance actually entered a terminating/terminated state.
    source "$REPO_DIR/.env"
    for i in $(seq 1 12); do
        STATUS=$(curl -s -u "$LAMBDA_API_KEY:" \
            "https://cloud.lambda.ai/api/v1/instances/$INSTANCE_ID" \
            | jq -r '.data.status // "missing"')
        echo "[cleanup] verify $i/12: instance status=$STATUS" >&2
        case "$STATUS" in
            terminating|terminated|missing)
                echo "[cleanup] OK -- instance $INSTANCE_ID confirmed $STATUS" >&2
                rm -f "$STATE_FILE"
                exit $code
                ;;
        esac
        sleep 5
    done
    echo "" >&2
    echo "*** [cleanup] WARNING: could not confirm instance $INSTANCE_ID terminated! ***" >&2
    echo "*** Check manually: ./scripts/lambda.sh instances ***" >&2
    echo "*** Force-terminate:  ./scripts/lambda.sh terminate $INSTANCE_ID ***" >&2
    echo "*** State file kept at $STATE_FILE for retry ***" >&2
    exit $code
}
trap cleanup EXIT INT TERM

# --- launch ---
echo "[lambda_train] launching..." >&2
INSTANCE_ID=$("$LAMBDA_CMD" launch "$INSTANCE_TYPE" "$REGION")
if [ -z "$INSTANCE_ID" ] || [ "$INSTANCE_ID" = "null" ]; then
    echo "[lambda_train] launch failed; check output above." >&2
    exit 1
fi
echo "$INSTANCE_ID" > "$STATE_FILE"
echo "[lambda_train] instance id: $INSTANCE_ID" >&2

# --- wait for SSH ---
IP=$("$LAMBDA_CMD" wait-ready "$INSTANCE_ID")
if [ -z "$IP" ]; then
    echo "[lambda_train] failed to get IP / SSH" >&2
    exit 1
fi
echo "[lambda_train] instance ready at $IP" >&2

# --- rsync repo ---
echo "[lambda_train] rsyncing repo to instance..." >&2
rsync -az --delete \
    --exclude '/.venv/' \
    --exclude '/.git/' \
    --exclude '/data/' \
    --exclude '/paper/' \
    --exclude '/.lambda-state/' \
    --exclude '__pycache__/' \
    --exclude '.pytest_cache/' \
    --exclude '*.pt' \
    -e "ssh -o StrictHostKeyChecking=accept-new" \
    "$REPO_DIR/" "ubuntu@$IP:/home/ubuntu/group-moe/"

# --- run training ---
# SSH keepalive (ServerAliveInterval/CountMax) prevents idle-timeout death during
# long quiet phases like torch-cluster compile (~10-15 min) and per-epoch waits.
# 60s keepalive × max 10 missed = SSH gives up only after 10 min of true silence.
# Forward selected env vars (FORCE_REBUILD_VENV) so callers can override remote behavior.
REMOTE_ENV=""
if [ -n "${FORCE_REBUILD_VENV:-}" ]; then
    REMOTE_ENV="$REMOTE_ENV FORCE_REBUILD_VENV=$FORCE_REBUILD_VENV"
fi
echo "[lambda_train] starting training (output streams below)..." >&2
echo "================================================================"
ssh -o StrictHostKeyChecking=accept-new \
    -o ServerAliveInterval=60 \
    -o ServerAliveCountMax=10 \
    ubuntu@"$IP" \
    "$REMOTE_ENV bash /home/ubuntu/group-moe/scripts/lambda_remote_setup.sh ${TRAIN_ARGS[*]}"
echo "================================================================"

# --- sync results back ---
LOCAL_RUN_DIR="$REPO_DIR/data/qm9/$RUN_NAME"
mkdir -p "$LOCAL_RUN_DIR"
echo "[lambda_train] rsyncing results to $LOCAL_RUN_DIR..." >&2
rsync -az -e "ssh -o StrictHostKeyChecking=accept-new" \
    "ubuntu@$IP:/lambda/nfs/group-moe-data/data/qm9/$RUN_NAME/" \
    "$LOCAL_RUN_DIR/" || echo "[lambda_train] WARNING: rsync of results failed" >&2

echo "[lambda_train] training complete." >&2
# trap will terminate the instance
