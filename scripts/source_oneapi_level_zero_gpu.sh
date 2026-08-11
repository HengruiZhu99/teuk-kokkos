#!/usr/bin/env bash

# Source this file, then call teuk_source_oneapi_level_zero_gpu. The function
# fails closed unless Intel oneAPI and a Level Zero GPU are both available.
teuk_source_oneapi_level_zero_gpu() {
  local setvars_path="${TEUK_ONEAPI_SETVARS:-/home/hzhu/intel/oneapi/setvars.sh}"
  local enumeration_log="${TEUK_SYCL_ENUMERATION_LOG:-/tmp/teuk-sycl-ls.txt}"

  if [[ ! -f "${setvars_path}" ]]; then
    echo "oneAPI setvars script not found: ${setvars_path}" >&2
    return 1
  fi

  local nounset_was_on=0
  [[ $- == *u* ]] && nounset_was_on=1 && set +u
  # shellcheck disable=SC1090
  source "${setvars_path}" --force
  (( nounset_was_on == 1 )) && set -u

  export ONEAPI_DEVICE_SELECTOR=level_zero:gpu
  command -v icpx >/dev/null || {
    echo "icpx is unavailable after oneAPI activation" >&2
    return 1
  }
  command -v sycl-ls >/dev/null || {
    echo "sycl-ls is unavailable after oneAPI activation" >&2
    return 1
  }

  sycl-ls >"${enumeration_log}" 2>&1 || {
    cat "${enumeration_log}" >&2
    return 1
  }
  cat "${enumeration_log}"
  if ! rg -q 'level_zero:gpu' "${enumeration_log}"; then
    echo "No Level Zero GPU was enumerated" >&2
    return 1
  fi
}

