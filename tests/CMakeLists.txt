set(RRI_TESTS
  test_gis_io
  test_hubeny
  test_section
  test_idx_setting
  test_kernels_zero_gradient
  test_storage_calc
)

foreach(t ${RRI_TESTS})
  add_executable(${t} ${t}.c)
  target_link_libraries(${t} PRIVATE rri_core)
  add_test(NAME ${t} COMMAND ${t})
endforeach()
