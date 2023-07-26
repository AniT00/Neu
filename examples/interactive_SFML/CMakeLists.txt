
if(CMAKE_BUILD_TYPE STREQUAL "Release")
	message(FATAL_ERROR "Can't build release.")
endif()

#add_executable(interactive interactive.cpp )

set(SRC interactive.cpp)

neu_add_example(interactive SOURCES ${SRC})

# SFML
set(SFML_DIRECTORY "C:/libs/SFML-2.6.0")

add_compile_definitions(SFML_STATIC)

target_include_directories(interactive PUBLIC "${SFML_DIRECTORY}/include")

target_link_directories(interactive PUBLIC "${SFML_DIRECTORY}/lib")

target_link_libraries(interactive 
sfml-graphics-s-d
sfml-system-s-d
sfml-window-s-d
opengl32
winmm
gdi32
)

# OpenCL
set(OPENCL_DIRECTORY "C:/libs/OpenCL")

target_include_directories(interactive PUBLIC "${OPENCL_DIRECTORY}/include")

target_link_directories(interactive PUBLIC "${OPENCL_DIRECTORY}/lib")

target_link_libraries(interactive OpenCL)
