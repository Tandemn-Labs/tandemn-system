#!/bin/bash

# Trap Ctrl+C and cleanup
cleanup() {
    echo ""
    echo "Stopping services..."
    kill $MR_PID 2>/dev/null || true
    kill $HA_PID 2>/dev/null || true
    wait $MR_PID 2>/dev/null
    wait $HA_PID 2>/dev/null
    echo "Services stopped"
    exit 0
}

trap cleanup SIGINT SIGTERM

python3 machine_runner.py &
MR_PID=$!

python3 node_health_agent.py &
HA_PID=$!

echo "Services started (PIDs: MR=$MR_PID, HA=$HA_PID)"
echo "Press Ctrl+C to stop"

# Wait for both processes
wait