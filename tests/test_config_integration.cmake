if(NOT DEFINED TEUK_SOLVER OR NOT DEFINED TEUK_INTEGRATION_DIR)
  message(FATAL_ERROR "integration test needs TEUK_SOLVER and TEUK_INTEGRATION_DIR")
endif()

file(REMOVE_RECURSE "${TEUK_INTEGRATION_DIR}")
file(MAKE_DIRECTORY "${TEUK_INTEGRATION_DIR}")

function(run_one name spin)
  set(output "${TEUK_INTEGRATION_DIR}/${name}-output")
  set(config "${TEUK_INTEGRATION_DIR}/${name}.cfg")
  file(WRITE "${config}"
"config_version = 1
mass = 1.0
spin = ${spin}
compactification_length = 1.0
nr = 9
ntheta = 6
ellmax_first = 3
ellmax_second = 3
first_order_modes = -1,1
second_order_modes = -1,1
final_time = 1e-6
steps = 1
cfl = 0.1
initial_data.type = gaussian
initial_data.seed_ell = 3
initial_data.seed_m = 1
initial_data.amplitude_real = 1e-6
initial_data.amplitude_imag = 0
second_order.enabled = false
output.directory = ${output}
output.diagnostic_every = 1
output.checkpoint_every = 0
")
  execute_process(
    COMMAND "${TEUK_SOLVER}" --config "${config}"
    RESULT_VARIABLE status
    OUTPUT_VARIABLE stdout
    ERROR_VARIABLE stderr)
  if(NOT status EQUAL 0)
    message(FATAL_ERROR "${name} failed: ${stderr}\n${stdout}")
  endif()
  set(resolved "${output}/resolved_config.cfg")
  if(NOT EXISTS "${resolved}")
    message(FATAL_ERROR "${name} did not write resolved_config.cfg")
  endif()
  if(NOT EXISTS "${output}/waveforms.csv")
    message(FATAL_ERROR "${name} did not write waveforms.csv")
  endif()
  file(READ "${resolved}" resolved_text)
  string(FIND "${resolved_text}" "spin = ${spin}" spin_position)
  if(spin_position EQUAL -1)
    message(FATAL_ERROR "${name} resolved spin is incorrect")
  endif()
  string(FIND "${resolved_text}" "plus2.enabled = false" plus2_position)
  if(plus2_position EQUAL -1)
    message(FATAL_ERROR "${name} did not resolve the disabled plus2 default")
  endif()
endfunction()

run_one(spin_zero 0)
run_one(spin_half 0.5)

file(READ "${TEUK_INTEGRATION_DIR}/spin_zero-output/resolved_config.cfg" zero)
file(READ "${TEUK_INTEGRATION_DIR}/spin_half-output/resolved_config.cfg" half)
if(zero STREQUAL half)
  message(FATAL_ERROR "two runs from one executable produced identical configs")
endif()

set(plus2_output "${TEUK_INTEGRATION_DIR}/plus2-rejected-output")
set(plus2_config "${TEUK_INTEGRATION_DIR}/plus2-rejected.cfg")
file(WRITE "${plus2_config}"
"config_version = 1
plus2.enabled = true
plus2.mode = diagnostic_only
output.directory = ${plus2_output}
")
execute_process(
  COMMAND "${TEUK_SOLVER}" --config "${plus2_config}"
  RESULT_VARIABLE plus2_status
  OUTPUT_VARIABLE plus2_stdout
  ERROR_VARIABLE plus2_stderr)
if(plus2_status EQUAL 0)
  message(FATAL_ERROR "enabled plus2 mode unexpectedly entered production")
endif()
string(FIND "${plus2_stderr}"
  "plus2 mode 'diagnostic_only' is not production integrated"
  plus2_error_position)
if(plus2_error_position EQUAL -1)
  message(FATAL_ERROR
    "enabled plus2 mode did not fail with the production gate: ${plus2_stderr}")
endif()
if(EXISTS "${plus2_output}")
  message(FATAL_ERROR "rejected plus2 mode allocated an output directory")
endif()
