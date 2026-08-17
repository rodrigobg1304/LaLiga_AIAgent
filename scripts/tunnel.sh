#!/bin/bash
# Túnel público bajo demanda para enseñar la API de predicción (localhost:8001)
# a alguien en persona. Manual a propósito: no se deja corriendo permanentemente,
# y la URL cambia cada vez que se arranca, así que un enlace viejo deja de servir
# en cuanto se para.
#
# Uso:
#   scripts/tunnel.sh start   -> arranca el túnel y muestra la URL pública
#   scripts/tunnel.sh stop    -> lo mata
#   scripts/tunnel.sh status  -> URL actual si está corriendo

set -euo pipefail
LOG_FILE="$(dirname "$0")/logs/cloudflared_tunnel.log"
PID_FILE="/tmp/laliga_cloudflared_tunnel.pid"

start() {
    if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
        echo "Ya está corriendo (PID $(cat "$PID_FILE")). URL:"
        grep -o "https://[a-zA-Z0-9.-]*trycloudflare.com" "$LOG_FILE" | tail -1
        exit 0
    fi
    if ! curl -s -o /dev/null -w "" localhost:8001/docs; then
        echo "Aviso: el servicio de predicción (localhost:8001) no responde. Arráncalo antes." >&2
    fi
    : > "$LOG_FILE"
    nohup cloudflared tunnel --url http://localhost:8001 > "$LOG_FILE" 2>&1 &
    echo $! > "$PID_FILE"
    disown
    echo "Arrancando túnel..."
    for _ in $(seq 1 15); do
        sleep 1
        url=$(grep -o "https://[a-zA-Z0-9.-]*trycloudflare.com" "$LOG_FILE" | head -1 || true)
        if [ -n "$url" ]; then
            echo "URL pública: $url"
            exit 0
        fi
    done
    echo "No se pudo obtener la URL a tiempo, revisa $LOG_FILE" >&2
    exit 1
}

stop() {
    if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
        kill "$(cat "$PID_FILE")"
        rm -f "$PID_FILE"
        echo "Túnel detenido."
    else
        pkill -f "cloudflared tunnel --url http://localhost:8001" 2>/dev/null && echo "Túnel detenido." || echo "No había túnel corriendo."
        rm -f "$PID_FILE"
    fi
}

status() {
    if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
        echo "Corriendo (PID $(cat "$PID_FILE")). URL:"
        grep -o "https://[a-zA-Z0-9.-]*trycloudflare.com" "$LOG_FILE" | tail -1
    else
        echo "Parado."
    fi
}

case "${1:-}" in
    start) start ;;
    stop) stop ;;
    status) status ;;
    *) echo "Uso: $0 {start|stop|status}"; exit 1 ;;
esac
