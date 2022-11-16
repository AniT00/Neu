# CMakeList.txt: файл проекта CMake верхнего уровня; проведите глобальную конфигурацию
# и включите сюда подпроекты.
#
cmake_minimum_required (VERSION 3.8)

project ("Neu")

# Включите подпроекты.
add_subdirectory ("src/Neu/")

add_subdirectory("test")
