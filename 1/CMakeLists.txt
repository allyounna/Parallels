project(1)

set(DTYPE "float" CACHE STRING "float")

if(DTYPE MATCHES "float")
    add_executable(hello float.cpp)
else()
    add_executable(hello double.cpp)
endif()

target_link_libraries(hello)