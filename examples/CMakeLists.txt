set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/examples)

include_directories(PUBLIC ${PROJECT_SOURCE_DIR}/include/Neu)

add_subdirectory(and_op)
add_subdirectory(interactive_SFML)
add_subdirectory(fisher_iris)