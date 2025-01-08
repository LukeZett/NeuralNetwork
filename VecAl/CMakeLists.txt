cmake_minimum_required (VERSION 3.10)

project(VecAl VERSION 0.0.1
	LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

add_library(VecAl
	"VectorTypes/VectorLike.h"
	"VectorTypes/Vector.h"
	"VectorTypes/Vector.cpp"
	"MatOps/Scalarfloat.h"
	"VectorTypes/VectorLike.cpp"
	"MatrixTypes/MatrixLike.h" 
	"VectorTypes/VectorView.h"
	"MatrixTypes/Matrix.h"
	"MatrixTypes/Matrix.cpp" 
	"MatrixTypes/MatrixView.h"
	"MatrixTypes/MatrixLike.cpp" "MatrixArrayTypes/MatrixArrayLike.h" "MatrixArrayTypes/MatrixArray.h" "MatrixArrayTypes/MatrixArrayView.h" "MatOps/AVX256float.h" "MatrixArrayTypes/MatrixArrayLike.cpp")

target_include_directories(VecAl PUBLIC ".")
target_include_directories(VecAl PUBLIC "MatrixTypes")
target_include_directories(VecAl PUBLIC "VectorTypes")
target_include_directories(VecAl PUBLIC "MatrixArrayTypes")

### Google tests ###

include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
)

# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

enable_testing()

add_executable(
	BasicTests
	"VectorTypes/VectorLike.h"
	"VectorTypes/Vector.h"
	"VectorTypes/Vector.cpp"
	"MatOps/Scalarfloat.h"
	"MatOps/AVX512float.h"
	"Gtests/VectorBasicTest.cpp"
	"VectorTypes/VectorLike.cpp"
	"MatrixTypes/MatrixLike.h"
	"VectorTypes/VectorView.h" "MatrixTypes/Matrix.h" "MatrixTypes/Matrix.cpp" "MatrixTypes/MatrixView.h" "MatrixTypes/MatrixLike.cpp" "Gtests/MatrixMoveTest.cpp" "Gtests/MatMulTest.cpp" "MatrixArrayTypes/MatrixArrayLike.h" "MatrixArrayTypes/MatrixArray.h" "MatrixArrayTypes/MatrixArrayView.h" "MatOps/AVX256float.h" "MatrixArrayTypes/MatrixArrayLike.cpp")

include_directories(BasicTests ".")
include_directories(BasicTests "VectorTypes")
include_directories(BasicTests "MatrixTypes")

target_link_libraries(
	BasicTests
	GTest::gtest_main
)

include(GoogleTest)
gtest_discover_tests(BasicTests)


if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
  # using Clang
	set(CMAKE_CXX_FLAGS "-mavx -O3 -pthread")
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  # using GCC
	set(CMAKE_CXX_FLAGS "-mavx -O3 -pthread")
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
  # using Intel C++
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
  # using Visual Studio C++
endif()