add_executable(test-main main.cpp)

target_link_libraries(test-main Neu)

target_include_directories(test-main PUBLIC ${PROJECT_SOURCE_DIR}/include/Neu)