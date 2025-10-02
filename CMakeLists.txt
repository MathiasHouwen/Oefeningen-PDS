cmake_minimum_required(VERSION 4.0)
project(Oefeningen_PDS)

set(CMAKE_CXX_STANDARD 23)

# Add MS-MPI include and library paths
include_directories("C:/Program Files (x86)/Microsoft SDKs/MPI/Include")
link_directories("C:/Program Files (x86)/Microsoft SDKs/MPI/Lib/x64")

add_executable(Oefeningen_PDS Oefening-8/oef8.cpp)

# Link against MS-MPI library
target_link_libraries(Oefeningen_PDS msmpi)