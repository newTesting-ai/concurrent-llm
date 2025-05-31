cmake_minimum_required(VERSION 3.12)
project(ThreadPool)

# Set C++ standard (llama.cpp requires C++11 minimum)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
find_package(Threads REQUIRED)

# Option 1: Add llama.cpp as subdirectory (if you have it locally)
# Uncomment these lines if llama.cpp is in a subdirectory
add_subdirectory(llama.cpp)
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)



# Create executable
add_executable(threadpool 
                main.cpp
                threadPool.cpp
                context_worker.cpp
                llama_schedular.cpp
)

# Link libraries
target_link_libraries(threadpool 
    PRIVATE
    Threads::Threads
    llama  # Main llama library
)

# Include llama.cpp headers
target_include_directories(threadpool 
    PRIVATE 
    ${llama_SOURCE_DIR}
    ${llama_SOURCE_DIR}/include
    ${llama_SOURCE_DIR}/common
)

# Enable warnings
target_compile_options(threadpool PRIVATE -Wall -Wextra)

# For debug builds
set(CMAKE_BUILD_TYPE Debug)
