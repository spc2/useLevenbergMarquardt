cmake_minimum_required(VERSION 3.13)
project(useLevenbergMarquardt)

set(CMAKE_BUILD_TYPE "Release" )
set(CMAKE_CXX_STANDARD 14)

list( APPEND CMAKE_MODULE_PATH /home/spc/Download/g2o/cmake_modules )

# OpenCV
find_package(OpenCV REQUIRED)
# find_package(MathGL2 REQUIRED)

include_directories(
        ${OpenCV_DIRS}
        # ${MathGL2_INCLUDE_DIRS}
        "/usr/include/eigen3/"
)

add_executable(useLevenbergMarquardt levmarq.cpp)
target_link_libraries(useLevenbergMarquardt
        ${OpenCV_LIBS}
        # ${MathGL2_LIBRARIES}
        )