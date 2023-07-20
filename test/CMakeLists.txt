add_executable(test main.cpp "main.cpp")

target_link_libraries(test Neu)

target_include_directories(test PUBLIC ${PROJECT_SOURCE_DIR}/include/Neu)

file(COPY "samples/Iris.csv" DESTINATION ${CMAKE_CURRENT_BINARY_DIR})