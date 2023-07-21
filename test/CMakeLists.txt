add_executable(test main.cpp "main.cpp")

set(SFML_DIRECTORY "C:/SFML-2.6.0")

target_include_directories(test PRIVATE "${SFML_DIRECTORY}/include")

target_link_directories(test PRIVATE "${SFML_DIRECTORY}/lib")

target_link_libraries(test sfml-graphics-d sfml-system-d sfml-window-d)

target_link_libraries(test Neu )

target_include_directories(test PUBLIC ${PROJECT_SOURCE_DIR}/include/Neu)

file(COPY "samples/Iris.csv" DESTINATION ${CMAKE_CURRENT_BINARY_DIR})