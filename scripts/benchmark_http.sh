#!/usr/bin/env bash
# Shared HTTP setup/readiness checks for locally launched benchmark servers.

configure_benchmark_http() {
    export PORT="${PORT:-30000}"
    export CLIENT_BASE_URL="${CLIENT_BASE_URL:-http://127.0.0.1:${PORT}}"
    CLIENT_BASE_URL="${CLIENT_BASE_URL%/}"
    # Keep external proxy settings, but bypass them for local probes and clients.
    export NO_PROXY="${NO_PROXY:+${NO_PROXY},}${no_proxy:+${no_proxy},}127.0.0.1,localhost,::1"
    export no_proxy="${NO_PROXY}"
}

wait_for_server() {
    local deadline=$((SECONDS + SERVER_START_TIMEOUT))
    local next_report=0 previous="" status="" code rc remaining probe_timeout
    local error_file="${CURRENT_DIR}/health_check.error"
    echo "Waiting for ${CLIENT_BASE_URL}/health (listen ${HOST}:${PORT}, timeout ${SERVER_START_TIMEOUT}s)"
    while (( SECONDS < deadline )); do
        if [[ -n "${SERVER_PID}" ]] && ! kill -0 "${SERVER_PID}" 2>/dev/null; then
            echo "Server exited before becoming healthy. See ${CURRENT_DIR}/server.log" >&2
            tail -n 40 "${CURRENT_DIR}/server.log" >&2 || true
            return 1
        fi
        remaining=$((deadline - SECONDS))
        probe_timeout=30
        if (( remaining < probe_timeout )); then probe_timeout=${remaining}; fi
        rc=0
        code=$(curl --noproxy "${NO_PROXY}" --silent --show-error \
            --connect-timeout 3 --max-time "${probe_timeout}" \
            --output /dev/null --write-out '%{http_code}' \
            "${CLIENT_BASE_URL}/health" 2>"${error_file}") || rc=$?
        if [[ "${rc}" == 0 && "${code}" == 200 ]]; then
            echo "Server ready: ${CLIENT_BASE_URL}/health returned HTTP 200"
            return 0
        fi
        status="HTTP ${code:-000}, curl exit ${rc}"
        if [[ "${status}" != "${previous}" ]] || (( SECONDS >= next_report )); then
            echo "Waiting for health: ${status}; remaining $((deadline - SECONDS))s" >&2
            if [[ -s "${error_file}" ]]; then cat "${error_file}" >&2; fi
            previous="${status}"
            next_report=$((SECONDS + 30))
        fi
        if (( SECONDS < deadline )); then sleep 2; fi
    done
    echo "Timed out waiting for ${CLIENT_BASE_URL}/health (${status}). See ${CURRENT_DIR}/server.log" >&2
    echo "HTTP 503 means the server is not ready; a listening port alone does not prove worker readiness." >&2
    tail -n 40 "${CURRENT_DIR}/server.log" >&2 || true
    return 1
}
