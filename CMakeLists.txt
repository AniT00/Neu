cmake_minimum_required (VERSION 3.8)

project ("Neu")

set (CMAKE_CXX_STANDARD 11)

# Включите подпроекты.
add_subdirectory ("src/Neu/")

add_subdirectory("test")
