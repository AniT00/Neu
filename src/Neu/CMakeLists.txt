# include the Neu specific macros
include(${PROJECT_SOURCE_DIR}/cmake/Macros.cmake)

set(INCROOT ${PROJECT_SOURCE_DIR}/include/Neu)
set(SRCROOT ${PROJECT_SOURCE_DIR}/src/Neu)

set(SRC
	${INCROOT}/NeuralNetwork.h
	${SRCROOT}/NeuralNetwork.cpp
	${INCROOT}/CsvReader.h
	${SRCROOT}/CsvReader.cpp
	${INCROOT}/Layer.h
	${SRCROOT}/Layer.cpp
	${INCROOT}/Activators.h
	${SRCROOT}/Activators.cpp
	${INCROOT}/Logger.h
	${SRCROOT}/Logger.cpp
	
	
)
source_group("" FILES ${SRC})

add_library(Neu STATIC ${SRC})

set_target_properties(Neu PROPERTIES FOLDER "Neu")

target_include_directories(Neu PUBLIC ${PROJECT_SOURCE_DIR}/include)

# CLBlast
set(CLBLAST_DIRECTORY "C:/Users/Ян/source/libs/CLBlast")
target_include_directories(Neu PUBLIC ${CLBLAST_DIRECTORY}/include)
target_link_directories(Neu PUBLIC ${CLBLAST_DIRECTORY}/lib)
target_link_libraries(Neu clblast)

# OpenCL
set(OPENCL_DIRECTORY "C:/libs/OpenCL")
target_include_directories(Neu PUBLIC "${OPENCL_DIRECTORY}/include")
target_link_directories(Neu PUBLIC "${OPENCL_DIRECTORY}/lib")
target_link_libraries(Neu OpenCL)
