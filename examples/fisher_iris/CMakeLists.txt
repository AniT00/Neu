set(SRC fisher_iris.cpp)

neu_add_example(fisher_iris SOURCES ${SRC})

add_custom_target(samples
  ${CMAKE_COMMAND} 
  -D
  SOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}/samples
  -D
  DESTINATION_DIR=${CMAKE_CURRENT_BINARY_DIR}
  -P
  ${PROJECT_SOURCE_DIR}/cmake/Copy.cmake)

add_dependencies(fisher_iris samples)