#!/usr/bin/env bash
# Source this file from benchmark runners before importing native Python modules.

configure_cpp_runtime() {
    local python_prefix selected entry merged="" inherited="${LD_PRELOAD:-}"
    local -a preload_entries
    python_prefix="$(python -c 'import sys; print(sys.prefix)')" || return
    # An explicit value (including empty) takes precedence over auto-detection.
    if [[ "${PRELOAD_LIBSTDCXX+x}" == x ]]; then
        selected="${PRELOAD_LIBSTDCXX}"
    elif [[ -f "${python_prefix}/lib/libstdc++.so.6" ]]; then
        selected="${python_prefix}/lib/libstdc++.so.6"
    else
        selected=""
    fi
    if [[ -n "${selected}" ]]; then
        if [[ ! -f "${selected}" ]]; then
            echo "C++ runtime does not exist: ${selected}" >&2
            return 1
        fi
        # Replace stale libstdc++ preloads, while retaining unrelated preloads.
        if [[ -n "${inherited}" ]]; then
            IFS=' ' read -r -a preload_entries <<< "${inherited//:/ }"
            for entry in "${preload_entries[@]}"; do
                case "${entry##*/}" in
                    libstdc++.so*) continue ;;
                esac
                merged="${merged:+${merged}:}${entry}"
            done
        fi
        export LD_PRELOAD="${selected}${merged:+:${merged}}"
        echo "C++ runtime: ${selected}"
    fi
    export PRELOAD_LIBSTDCXX="${selected}"
}

check_zmq_runtime() {
    if ! python -c 'import sys, zmq; print("ZeroMQ import OK:", sys.executable, "libzmq", zmq.zmq_version())'; then
        echo "ZeroMQ import failed before model startup." >&2
        echo "For GLIBCXX errors, check the selected libstdc++ and update libstdcxx-ng in this Python environment; GLIBCXX_3.4.29 requires a GCC 11.1+ runtime." >&2
        echo "Example (in the active Conda environment): conda install 'libstdcxx-ng>=11.1'" >&2
        echo "Then unset PRELOAD_LIBSTDCXX to retry automatic selection, or set it to a compatible libstdc++.so.6 path." >&2
        return 1
    fi
}
