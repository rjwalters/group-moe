#!/bin/bash
# Lambda Labs CLI helper for group-moe.
# Adapted from sibling ml-audio-codecs project.
#
# Usage:
#   ./scripts/lambda.sh instances              List running instances
#   ./scripts/lambda.sh types                  List available instance types with capacity
#   ./scripts/lambda.sh filesystems            List filesystems
#   ./scripts/lambda.sh ssh <id>               SSH to instance
#   ./scripts/lambda.sh ip <id>                Print instance IP
#   ./scripts/lambda.sh launch <type> [region] Launch instance with group-moe-data filesystem
#   ./scripts/lambda.sh terminate <id>         Terminate instance

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

if [ ! -f "$REPO_DIR/.env" ]; then
    echo "ERROR: $REPO_DIR/.env not found. Copy from sibling project or create with LAMBDA_API_KEY=..." >&2
    exit 1
fi
source "$REPO_DIR/.env"

if [ -z "$LAMBDA_API_KEY" ]; then
    echo "ERROR: LAMBDA_API_KEY not set in .env" >&2
    exit 1
fi

API_BASE="https://cloud.lambda.ai/api/v1"
SSH_KEY_NAME="${LAMBDA_SSH_KEY:-mac-studio}"
FS_NAME="${LAMBDA_FS_NAME-group-moe-data}"  # `-` not `:-` so LAMBDA_FS_NAME= (empty) means "no filesystem"
DEFAULT_REGION="${LAMBDA_REGION:-us-west-1}"

lambda_api() {
    curl -s -u "$LAMBDA_API_KEY:" "$API_BASE/$1"
}

case "${1:-help}" in
    instances|i)
        echo "=== Running Instances ==="
        lambda_api "instances" | jq -r '.data[] | "\(.id) | \(.instance_type.name) | \(.ip_address // "pending") | \(.status)"' 2>/dev/null
        ;;

    types|t)
        echo "=== Available Instance Types (with capacity) ==="
        lambda_api "instance-types" \
            | jq -r '.data | to_entries[] | select(.value.regions_with_capacity_available | length > 0) | "\(.key): $\((.value.instance_type.price_cents_per_hour/100))/hr - \(.value.regions_with_capacity_available | map(.name) | join(", "))"'
        ;;

    filesystems|fs)
        echo "=== Filesystems ==="
        lambda_api "file-systems" | jq -r '.data[] | "\(.id) | \(.name) | \(.region.name) | \((.bytes_used // 0)/1e9 | floor)GB used | in_use=\(.is_in_use)"'
        ;;

    ip)
        if [ -z "$2" ]; then echo "Usage: $0 ip <instance-id>" >&2; exit 1; fi
        lambda_api "instances/$2" | jq -r '.data.ip_address // "pending"'
        ;;

    ssh)
        if [ -z "$2" ]; then echo "Usage: $0 ssh <instance-id>" >&2; exit 1; fi
        IP=$(lambda_api "instances/$2" | jq -r '.data.ip_address')
        if [ "$IP" = "null" ] || [ -z "$IP" ]; then
            echo "Instance $2 has no IP yet (still launching?)" >&2
            exit 1
        fi
        echo "Connecting to ubuntu@$IP..."
        exec ssh -o StrictHostKeyChecking=accept-new ubuntu@"$IP"
        ;;

    launch)
        if [ -z "$2" ]; then
            echo "Usage: $0 launch <instance-type> [region]" >&2
            echo "Example: $0 launch gpu_1x_a100_sxm4 us-west-1" >&2
            exit 1
        fi
        REGION="${3:-$DEFAULT_REGION}"

        if [ -n "$FS_NAME" ]; then
            echo "Launching $2 in $REGION with filesystem $FS_NAME..." >&2
            PAYLOAD=$(jq -n \
                --arg region "$REGION" \
                --arg type "$2" \
                --arg ssh_key "$SSH_KEY_NAME" \
                --arg fs "$FS_NAME" \
                '{
                    region_name: $region,
                    instance_type_name: $type,
                    ssh_key_names: [$ssh_key],
                    file_system_names: [$fs],
                    name: "group-moe-training"
                }')
        else
            echo "Launching $2 in $REGION (no filesystem)..." >&2
            PAYLOAD=$(jq -n \
                --arg region "$REGION" \
                --arg type "$2" \
                --arg ssh_key "$SSH_KEY_NAME" \
                '{
                    region_name: $region,
                    instance_type_name: $type,
                    ssh_key_names: [$ssh_key],
                    name: "group-moe-training"
                }')
        fi
        RESPONSE=$(curl -s -u "$LAMBDA_API_KEY:" -X POST "$API_BASE/instance-operations/launch" \
            -H "Content-Type: application/json" \
            -d "$PAYLOAD")
        # Print the API response to stderr (humans see it; stdout reserved for the ID).
        echo "$RESPONSE" | jq . >&2
        ID=$(echo "$RESPONSE" | jq -r '.data.instance_ids[0] // empty')
        if [ -z "$ID" ]; then
            echo "[lambda.sh] launch failed: no instance ID in response" >&2
            exit 1
        fi
        echo "$ID"
        ;;

    terminate)
        if [ -z "$2" ]; then echo "Usage: $0 terminate <instance-id>" >&2; exit 1; fi
        echo "Terminating $2..." >&2
        curl -s -u "$LAMBDA_API_KEY:" -X POST "$API_BASE/instance-operations/terminate" \
            -H "Content-Type: application/json" \
            -d "{\"instance_ids\": [\"$2\"]}" | jq .
        ;;

    wait-ready)
        # Block until the instance has an IP and SSH responds.
        # Lambda populates `ip_address` and `hostname` at different times — `hostname`
        # (in dash-form like 64-181-240-218) often appears first, so fall back to it.
        if [ -z "$2" ]; then echo "Usage: $0 wait-ready <instance-id>" >&2; exit 1; fi
        ID="$2"
        echo "[wait-ready] polling instance $ID for IP/hostname..." >&2
        IP=""
        for i in $(seq 1 90); do
            INFO=$(lambda_api "instances/$ID")
            STATUS=$(echo "$INFO" | jq -r '.data.status')
            IP=$(echo "$INFO" | jq -r '.data.ip_address // empty')
            HOSTNAME=$(echo "$INFO" | jq -r '.data.hostname // empty')
            # Convert dash-form hostname to IP (e.g. 64-181-240-218 -> 64.181.240.218)
            if [ -z "$IP" ] && [ -n "$HOSTNAME" ]; then
                IP=$(echo "$HOSTNAME" | tr '-' '.')
            fi
            echo "[wait-ready] poll $i/90: status=$STATUS ip=${IP:-<none>} hostname=${HOSTNAME:-<none>}" >&2
            if [ -n "$IP" ] && [ "$STATUS" = "active" ]; then break; fi
            if [ "$STATUS" = "terminated" ] || [ "$STATUS" = "terminating" ]; then
                echo "[wait-ready] instance entered $STATUS state, aborting" >&2
                exit 1
            fi
            sleep 10
        done
        if [ -z "$IP" ]; then echo "[wait-ready] timed out waiting for IP" >&2; exit 1; fi
        echo "[wait-ready] IP=$IP, waiting for SSH..." >&2
        for i in $(seq 1 60); do
            if ssh -o StrictHostKeyChecking=accept-new -o ConnectTimeout=5 -o BatchMode=yes \
                ubuntu@"$IP" 'echo ok' >/dev/null 2>&1; then
                echo "[wait-ready] SSH ready at $IP" >&2
                echo "$IP"
                exit 0
            fi
            echo "[wait-ready] ssh poll $i/60: not yet" >&2
            sleep 5
        done
        echo "[wait-ready] timed out waiting for SSH" >&2
        exit 1
        ;;

    *)
        echo "Lambda Labs CLI Helper (group-moe)"
        echo ""
        echo "Usage: $0 <command>"
        echo ""
        echo "Commands:"
        echo "  instances, i             List running instances"
        echo "  types, t                 List instance types with capacity"
        echo "  filesystems, fs          List filesystems"
        echo "  ip <id>                  Print instance IP"
        echo "  ssh <id>                 SSH to instance"
        echo "  launch <type> [region]   Launch instance (default region: $DEFAULT_REGION)"
        echo "  wait-ready <id>          Block until instance has IP and SSH responds"
        echo "  terminate <id>           Terminate instance"
        echo ""
        echo "Defaults (override via env):"
        echo "  LAMBDA_SSH_KEY=$SSH_KEY_NAME"
        echo "  LAMBDA_FS_NAME=$FS_NAME"
        echo "  LAMBDA_REGION=$DEFAULT_REGION"
        ;;
esac
