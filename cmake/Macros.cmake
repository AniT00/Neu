macro(neu_add_example target)
    cmake_parse_arguments(THIS "" "" "SOURCES" ${ARGN})

	add_executable(${target} ${THIS_SOURCES})
	set_target_properties(${target} PROPERTIES FOLDER "Examples" RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/")
	target_link_libraries(${target} Neu)
endmacro()

macro(neu_copy target)
	cmake_parse_arguments(THIS, "" "SOURCE;TARGETDIR" ${ARGN})
endmacro()