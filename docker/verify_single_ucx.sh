#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Fail closed if NIXL can see a UCX installation other than the image's
# canonical /opt/hpcx/ucx tree. Pass a PID to also inspect loaded libraries.
set -euo pipefail

expected_root="${EXPECTED_UCX_ROOT:-/opt/hpcx/ucx}"
expected_version="${EXPECTED_UCX_VERSION:-1.21}"

fail() {
    echo "single-UCX verification failed: $*" >&2
    exit 1
}

test -x "${expected_root}/bin/ucx_info" \
    || fail "${expected_root}/bin/ucx_info is missing"

ucx_version="$("${expected_root}/bin/ucx_info" -v \
    | sed -n 's/^# Library version: //p' | head -n 1)"
case "${ucx_version}" in
    "${expected_version}".*|"${expected_version}") ;;
    *) fail "UCX ${ucx_version:-unknown} is installed; expected ${expected_version}.x" ;;
esac

if test -e /usr/local/ucx; then
    test "$(readlink -f /usr/local/ucx)" = "$(readlink -f "${expected_root}")" \
        || fail "/usr/local/ucx does not resolve to ${expected_root}"
fi

# A repaired PyPI wheel typically exposes the accidental second copy through
# hashed names such as nixl_cu13.libs/libucs-<hash>.so.0.0.0.
duplicate_ucx="$({
    for root in /usr /opt /workspace; do
        test -e "${root}" || continue
        find "${root}" -xdev \( -type f -o -type l \) \( \
            -name 'libucm.so*' -o -name 'libucm-*.so*' -o -name 'libucm_*.so*' -o \
            -name 'libucp.so*' -o -name 'libucp-*.so*' -o -name 'libucp_*.so*' -o \
            -name 'libucs.so*' -o -name 'libucs-*.so*' -o -name 'libucs_*.so*' -o \
            -name 'libuct.so*' -o -name 'libuct-*.so*' -o -name 'libuct_*.so*' \
        \) -print 2>/dev/null
    done
} | while IFS= read -r path; do
    resolved="$(readlink -f "${path}" 2>/dev/null || printf '%s' "${path}")"
    case "${resolved}" in
        "${expected_root}"/*) ;;
        *) printf '%s\n' "${path}" ;;
    esac
done | sort -u)"
test -z "${duplicate_ucx}" \
    || fail "UCX libraries exist outside ${expected_root}:\n${duplicate_ucx}"

plugins="$(find /usr/local/lib /opt/nvidia -type f -name 'libplugin_UCX.so' \
    -print 2>/dev/null | sort -u)"
test -n "${plugins}" || fail "NIXL UCX plugin was not found"

while IFS= read -r plugin; do
    readelf -d "${plugin}" | grep -Fq "${expected_root}/lib" \
        || fail "${plugin} has no ${expected_root}/lib RPATH/RUNPATH"

    bad_links="$(LD_LIBRARY_PATH="${expected_root}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" \
        ldd "${plugin}" \
        | awk '/libuc[mpst]([_.-]|\.so)/ && $0 !~ /\/opt\/hpcx\/ucx\/lib\// {print}')"
    test -z "${bad_links}" \
        || fail "${plugin} resolves UCX outside ${expected_root}:\n${bad_links}"
done <<< "${plugins}"

if test "$#" -gt 1; then
    fail "usage: $0 [pid]"
fi

if test "$#" -eq 1; then
    pid="$1"
    test -r "/proc/${pid}/maps" || fail "cannot read /proc/${pid}/maps"
    loaded_ucx="$(grep -Eo '/[^ ]*/libuc[mpst]([_.-][^ /]+)?\.so[^ ]*' \
        "/proc/${pid}/maps" | sort -u || true)"
    test -n "${loaded_ucx}" || fail "PID ${pid} has not loaded UCX"

    bad_maps="$(printf '%s\n' "${loaded_ucx}" \
        | awk '$0 !~ /^\/opt\/hpcx\/ucx\/lib\// {print}')"
    test -z "${bad_maps}" \
        || fail "PID ${pid} loaded UCX outside ${expected_root}:\n${bad_maps}"
fi

echo "single-UCX verification passed: UCX ${ucx_version}; plugin(s):"
while IFS= read -r plugin; do
    printf '  %s\n' "${plugin}"
done <<< "${plugins}"
