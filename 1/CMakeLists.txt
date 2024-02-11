project(1)

set(DTYPE "float" CACHE STRING "float")

if(DTYPE MATCHES "float")
    add_executable(sum float.cpp)
else()
    add_executable(sum double.cpp)
endif()

target_link_libraries(sum)
