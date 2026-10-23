#!/usr/bin/env bash
# ==============================================================================
# Jean SRE - Container Watchdog Daemon
# ==============================================================================
# Native background script to stream Docker events (event=die, event=health_status=unhealthy)
# and alert Jean SRE when critical fleet containers crash or enter unhealthy states.
# ==============================================================================

NODE_NAME=$(hostname)
AGENT_URL=${JEAN_AGENT_URL:-"http://localhost:3005"}
ALERT_COOLDOWN=300 # 5 minutes cooldown between alerts per container

# Temporary file to track alert timestamps per container
STATE_FILE="/tmp/container_watchdog_state"
touch "$STATE_FILE"

echo "[Container Watchdog] Initialized for node ${NODE_NAME} (Harness: ${AGENT_URL})"
echo "[Container Watchdog] Streaming Docker daemon events..."

# Send alert to Jean SRE
send_alert() {
    local name=$1
    local action=$2
    local safe_name=${name//[^a-zA-Z0-9_.-]/}
    local safe_action=${action//[^a-zA-Z0-9_.:-]/}
    local safe_node=${NODE_NAME//[^a-zA-Z0-9_.-]/}
    local desc="Docker event: Container '${safe_name}' triggered event '${safe_action}' on node ${safe_node}."
    
    echo "[Container Watchdog] 🚨 CONTAINER ANOMALY: ${safe_name} triggered ${safe_action}! Alerting Jean SRE..."
    
    curl -s -X POST "${AGENT_URL}/trigger" \
        -H "Content-Type: application/json" \
        -d "{\"type\":\"container_crash\",\"node\":\"${safe_node}\",\"description\":\"${desc}\"}" \
        -o /dev/null
}

# Stream events from Docker
docker events --filter 'event=die' --filter 'event=health_status=unhealthy' --format '{{.Actor.Attributes.name}} {{.Action}}' | while read -r name action; do
    if [ -n "$name" ] && [ -n "$action" ]; then
        safe_name=${name//[^a-zA-Z0-9_.-]/}
        CURRENT_TIME=$(date +%s)
        
        # Check cooldown state
        LAST_ALERT=0
        if grep -Fq "${safe_name} " "$STATE_FILE" 2>/dev/null; then
            LAST_ALERT=$(awk -v c="$safe_name" '$1 == c {print $2; exit}' "$STATE_FILE")
        fi
        
        if [ $((CURRENT_TIME - LAST_ALERT)) -ge $ALERT_COOLDOWN ]; then
            send_alert "$safe_name" "$action"
            
            # Update state file (upsert container timestamp)
            awk -v c="$safe_name" '$1 != c' "$STATE_FILE" > "${STATE_FILE}.tmp" 2>/dev/null
            echo "${safe_name} ${CURRENT_TIME}" >> "${STATE_FILE}.tmp"
            mv "${STATE_FILE}.tmp" "$STATE_FILE"
        else
            echo "[Container Watchdog] Ignored duplicate event for '${safe_name}' (cooldown active)"
        fi
    fi
done
