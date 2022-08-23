# Copyright (c) the JPEG XL Project
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

if((SANITIZER STREQUAL "asan") OR (SANITIZER STREQUAL "msan"))
  set(BUILD_TESTING OFF)
endif()

# Highway
set(HWY_SYSTEM_GTEST ON CACHE INTERNAL "")
set(HWY_FORCE_STATIC_LIBS ON CACHE INTERNAL "")
set(HWY_ENABLE_CONTRIB OFF CACHE INTERNAL "")
set(HWY_ENABLE_EXAMPLES OFF CACHE INTERNAL "")
if((SANITIZER STREQUAL "asan") OR (SANITIZER STREQUAL "msan"))
  set(HWY_ENABLE_INSTALL OFF CACHE INTERNAL "")
endif()
if (EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/highway/CMakeLists.txt" AND
    NOT JPEGXL_FORCE_SYSTEM_HWY)
  add_subdirectory(highway)
  configure_file("${CMAKE_CURRENT_SOURCE_DIR}/highway/LICENSE"
                 ${PROJECT_BINARY_DIR}/LICENSE.highway COPYONLY)
else()
  find_package(HWY 0.15.0)
  if (NOT HWY_FOUND)
    message(FATAL_ERROR
        "Highway library (hwy) not found. Install libhwy-dev or download it "
        "to third_party/highway from https://github.com/google/highway . "
        "Highway is required to build JPEG XL. You can run "
        "${PROJECT_SOURCE_DIR}/deps.sh to download this dependency.")
  endif()
  if(JPEGXL_DEP_LICENSE_DIR)
    configure_file("${JPEGXL_DEP_LICENSE_DIR}/libhwy-dev/copyright"
                   ${PROJECT_BINARY_DIR}/LICENSE.highway COPYONLY)
  endif()  # JPEGXL_DEP_LICENSE_DIR
endif()

