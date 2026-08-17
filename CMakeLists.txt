cmake_minimum_required(VERSION 3.17)
project("inference-gpu-test")

# =====================================================================
# DEPENDENCIES (self-contained SDK folder shipped with the repo)
# =====================================================================
set(SDK_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/SDK" CACHE PATH "SDK dependencies")

set(OpenCV_DIR "${SDK_ROOT}/opencv/build")
find_package(OpenCV REQUIRED)

set(FMT_ROOT "${SDK_ROOT}/fmt-11.1.4")

# =====================================================================
# TARGET
# =====================================================================
# IPC_Shared.h lives next to the source (option B: a byte-identical copy of the
# inference_gpu contract), so "#include "IPC_Shared.h"" resolves relative to the
# source file without any extra include dir.
add_executable(inference-gpu-test "inference_gpu_test.cpp")

# UNICODE build: MUST match inference_gpu / ONNX_inference so TCHAR == wchar_t and
# the MMF struct layout (pathModello[512] etc.) is identical on both sides. It
# also makes the Win32 calls resolve to the wide (...W) versions used here.
target_compile_definitions(inference-gpu-test PRIVATE UNICODE _UNICODE NOMINMAX)

set_property(TARGET inference-gpu-test PROPERTY CXX_STANDARD 20)          # <semaphore>
set_property(TARGET inference-gpu-test PROPERTY CXX_STANDARD_REQUIRED ON) # force /std:c++20

if (MSVC)
    target_compile_options(inference-gpu-test PRIVATE /utf-8)
endif()

target_include_directories(inference-gpu-test PRIVATE
    ${OpenCV_INCLUDE_DIRS}
    "${FMT_ROOT}/include")

target_link_libraries(inference-gpu-test
    ${OpenCV_LIBS}
    debug     "${FMT_ROOT}/build/lib/Debug/fmtd.lib"
    optimized "${FMT_ROOT}/build/lib/Release/fmt.lib")

# =====================================================================
# POST-BUILD: copy the OpenCV runtime DLLs next to the exe
# =====================================================================
add_custom_command(TARGET inference-gpu-test POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
    "${OpenCV_DIR}/x64/vc16/bin" "$<TARGET_FILE_DIR:inference-gpu-test>"
    COMMENT "Copying OpenCV DLLs to output directory...")
