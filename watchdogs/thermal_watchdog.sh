#!/usr/bin/env bash
# ==============================================================================
# Jean SRE - Thermal Watchdog Daemon
# ==============================================================================
# Native background script to monitor CPU and GPU thermal conditions and alert 
# the Jean SRE harness of emergency limits.
#
# Thresholds:
#   CPU: Warning at 85°C, Emergency at 95°C
#   GPU: Warning at 80°C, Emergency at 85°C
# ==============================================================================

NODE_NAME=$(hostname)
AGENT_URL=${JEAN_AGENT_URL:-"http://localhost:3005"}
POLL_INTERVAL=15
ALERT_COOLDOWN=300 # 5 minutes cooldown between alerts to prevent spam

LAST_CPU_ALERT=0
LAST_GPU_ALERT=0

echo "[Thermal Watchdog] Initialized for node ${NODE_NAME} (Harness: ${AGENT_URL})"
echo "[Thermal Watchdog] CPU Emergency: 95°C, GPU Emergency: 85°C"

# Get current maximum CPU temperature across all zones
get_cpu_temp() {
    local max_temp=0
    for zone in /sys/class/thermal/thermal_zone*/temp; do
        if [ -f "$zone" ]; then
            local temp_raw=$(cat "$zone")
            local temp=$((temp_raw / 1000))
            if [ "$temp" -gt "$max_temp" ]; then
                max_temp=$temp
            fi
        fi
    done
    echo "$max_temp"
}

# Get current maximum GPU temperature via nvidia-smi (if available)
get_gpu_temp() {
    if command -v nvidia-smi &> /dev/null; then
        local temp=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits 2>/dev/null | sort -rn | head -n1)
        if [ -n "$temp" ]; then
            echo "$temp"
            return
        fi
    fi
    echo "0"
}

# Send alert to Jean SRE
send_alert() {
    local type=${1//[^a-zA-Z0-9_-]/}
    local temp=${2//[^0-9.]/}
    local threshold=${3//[^0-9.]/}
    local safe_node=${NODE_NAME//[^a-zA-Z0-9_.-]/}
    local desc="Thermal emergency: ${type} temperature is ${temp}°C, exceeding safe threshold of ${threshold}°C on node ${safe_node}."
    
    echo "[Thermal Watchdog] 🚨 EXCEEDED LIMIT: ${type} is ${temp}°C! Alerting Jean SRE..."
    
    curl -s -X POST "${AGENT_URL}/trigger" \
        -H "Content-Type: application/json" \
        -d "{\"type\":\"thermal\",\"node\":\"${safe_node}\",\"description\":\"${desc}\"}" \
        -o /dev/null
}

# Main Loop
while true; do
    CURRENT_TIME=$(date +%s)
    
    # ── CPU Check ──
    CPU_TEMP=$(get_cpu_temp)
    if [ "$CPU_TEMP" -ge 95 ]; then
        if [ $((CURRENT_TIME - LAST_CPU_ALERT)) -ge $ALERT_COOLDOWN ]; then
            send_alert "CPU" "$CPU_TEMP" "95"
            LAST_CPU_ALERT=$CURRENT_TIME
        fi
    elif [ "$CPU_TEMP" -ge 85 ]; then
        echo "[Thermal Watchdog] [INFO] CPU Temp is warm: ${CPU_TEMP}°C"
    fi

    # ── GPU Check ──
    GPU_TEMP=$(get_gpu_temp)
    if [ "$GPU_TEMP" -ge 85 ]; then
        if [ $((CURRENT_TIME - LAST_GPU_ALERT)) -ge $ALERT_COOLDOWN ]; then
            send_alert "GPU" "$GPU_TEMP" "85"
            LAST_GPU_ALERT=$CURRENT_TIME
        fi
    elif [ "$GPU_TEMP" -ge 80 ]; then
        echo "[Thermal Watchdog] [INFO] GPU Temp is warm: ${GPU_TEMP}°C"
    fi

    sleep $POLL_INTERVAL
done
