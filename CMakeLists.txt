cmake_minimum_required (VERSION 3.8)

project ("Neu")

set (CMAKE_CXX_STANDARD 11)

set( CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin )

if (MSVC)
set_target_properties(${BUILD_TARGET} PROPERTIES LINK_FLAGS "/PROFILE")
endif()

add_subdirectory ("src/Neu/")

add_subdirectory("test")
