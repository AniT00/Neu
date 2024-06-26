set(INCROOT ${PROJECT_SOURCE_DIR}/include/gui)
set(SRCROOT ${PROJECT_SOURCE_DIR}/src/gui)

set(SRC
	main.cpp
	QtMainWindowClass.cpp
	QtMainWindowClass.h
	QtMainWindowClass.ui
)
source_group("" FILES ${SRC})

find_package(Qt6 REQUIRED COMPONENTS Widgets Charts Concurrent)
qt_standard_project_setup()

set(CMAKE_AUTOUIC ON)

add_executable(gui ${SRC})

#if($<CONFIG:Debug>)
#    get_target_property(WIDGETDLL Qt6::Widgets IMPORTED_LOCATION_DEBUG)
#else($<CONFIG:Debug>)
#    get_target_property(WIDGETDLL Qt6::Widgets IMPORTED_LOCATION_RELEASE)
#endif($<CONFIG:Debug>)
add_custom_command(
    TARGET gui POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        $<TARGET_FILE:Qt6::Widgets>
        $<TARGET_FILE:Qt6::Charts>
        $<TARGET_FILE:Qt6::Concurrent>
        $<TARGET_FILE:Qt6::Gui>
        $<TARGET_FILE:Qt6::Core>
        $<TARGET_FILE_DIR:gui>
)

add_custom_command(TARGET gui POST_BUILD 
    COMMAND Qt6::windeployqt
    ARGS $<TARGET_FILE:gui>
)

target_link_libraries(gui PRIVATE Qt6::Widgets Qt6::Charts Qt6::Concurrent Neu)

set_target_properties(gui PROPERTIES 
	FOLDER "gui"
	WIN32_EXECUTABLE ON
    MACOSX_BUNDLE ON)

target_include_directories(gui PUBLIC ${PROJECT_SOURCE_DIR}/include)

