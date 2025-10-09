# cmake_minimum_required(VERSION 4.0)
# project(Oefeningen_PDS)
#
# set(CMAKE_CXX_STANDARD 23)
#
# # Add MS-MPI include and library paths
# include_directories("C:/Program Files (x86)/Microsoft SDKs/MPI/Include")
# link_directories("C:/Program Files (x86)/Microsoft SDKs/MPI/Lib/x64")
#
# add_executable(Oefeningen_PDS Oefening-11/Oef11.cpp)
#
# # Link against MS-MPI library
# target_link_libraries(Oefeningen_PDS msmpi)

cmake_minimum_required(VERSION 3.21)
project(Oefeningen_PDS)

set(CMAKE_CXX_STANDARD 23)

# Find MS-MPI package (MSVC compatible)
find_package(MPI REQUIRED)

# Add your executable
add_executable(Oefeningen_PDS Oefening-12/Oef12.cpp)

# Link against MPI automatically
target_link_libraries(Oefeningen_PDS PRIVATE MPI::MPI_CXX)