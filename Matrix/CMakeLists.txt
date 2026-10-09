add_library(Matrix STATIC
    src/TypeDefine.cpp
    include/TypeDefine.h
)

target_include_directories(Matrix PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

if(MSVC)
    target_compile_options(Matrix PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/arch:AVX2>)
else()
    target_compile_options(Matrix PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-mavx2>)
endif()

find_package(CUDAToolkit REQUIRED)   # 保险，子目录再找一次
target_link_libraries(Matrix PUBLIC OpenMP::OpenMP_CXX CUDA::cudart)