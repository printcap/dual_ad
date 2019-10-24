cmake_minimum_required(VERSION 3.12)
project(dual_ad LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
enable_testing()

add_library(catch2 INTERFACE)
target_include_directories(catch2 INTERFACE include)

add_executable(dual_ad src/dual.cc)

add_executable(test_ad test/test_ad.cc)
target_link_libraries(test_ad PRIVATE catch2)
add_test(NAME test_ad COMMAND ./test_ad)
