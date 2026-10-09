add_executable(MatrixTest 
include/Col2ImTest.h
include/Dispatchers.h
include/Im2ColTest.h
include/MatmulTest.h
include/TestCommon.h
src/main.cpp
)
target_include_directories(MatrixTest PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(MatrixTest PRIVATE Matrix)