macro(neu_add_example target)
    cmake_parse_arguments(THIS "GUI_APP" "RESOURCES_DIR" "SOURCES;BUNDLE_RESOURCES;DEPENDS" ${ARGN})

	message(${target} ${THIS_SOURCES})
	add_executable(${target} ${THIS_SOURCES})
	set_target_properties(${target} PROPERTIES FOLDER "Examples")
	target_link_libraries(${target} Neu)
	set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
endmacro()