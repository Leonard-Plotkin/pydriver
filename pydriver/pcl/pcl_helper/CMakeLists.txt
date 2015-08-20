MACRO(ADD_PRECOMPILED_HEADER PrecompiledHeader PrecompiledSource SourcesVar)
  IF(MSVC)
    GET_FILENAME_COMPONENT(PrecompiledBasename ${PrecompiledHeader} NAME_WE)
    SET(PrecompiledBinary "${CMAKE_CURRENT_BINARY_DIR}/${PrecompiledBasename}.pch")
    SET(Sources ${${SourcesVar}})

    SET_SOURCE_FILES_PROPERTIES(${PrecompiledSource}
                                PROPERTIES COMPILE_FLAGS "/Yc\"${PrecompiledHeader}\" /Fp\"${PrecompiledBinary}\""
                                           OBJECT_OUTPUTS "${PrecompiledBinary}")
    SET_SOURCE_FILES_PROPERTIES(${Sources}
                                PROPERTIES COMPILE_FLAGS "/Yu\"${PrecompiledHeader}\" /FI\"${PrecompiledHeader}\" /Fp\"${PrecompiledBinary}\""
                                           OBJECT_DEPENDS "${PrecompiledBinary}")
  ENDIF(MSVC)
  # add precompiled header to SourcesVar
  LIST(APPEND ${SourcesVar} ${PrecompiledSource})
  LIST(APPEND ${SourcesVar} ${PrecompiledHeader})
ENDMACRO(ADD_PRECOMPILED_HEADER)

cmake_minimum_required(VERSION 2.6 FATAL_ERROR)
project(pcl_helper)
find_package(PCL 1.7.1 REQUIRED COMPONENTS common features filters geometry io keypoints search segmentation surface visualization)
include_directories(${PCL_INCLUDE_DIRS})
include_directories("../../common")
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

# raise memory limit for MSVC precompiled headers
IF(MSVC)
  SET(MSVC_COMPILE_FLAGS "/Zm500")
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${MSVC_COMPILE_FLAGS}")
ENDIF(MSVC)

# set output directory to the parent directory
set(outputdirectory "../lib")
# first for the generic no-config case (e.g. with mingw)
set( CMAKE_RUNTIME_OUTPUT_DIRECTORY ${outputdirectory} )
set( CMAKE_LIBRARY_OUTPUT_DIRECTORY ${outputdirectory} )
set( CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${outputdirectory} )
# second, for multi-config builds (e.g. msvc)
foreach( OUTPUTCONFIG ${CMAKE_CONFIGURATION_TYPES} )
    string( TOUPPER ${OUTPUTCONFIG} OUTPUTCONFIG )
    set( CMAKE_RUNTIME_OUTPUT_DIRECTORY_${OUTPUTCONFIG} ${outputdirectory} )
    set( CMAKE_LIBRARY_OUTPUT_DIRECTORY_${OUTPUTCONFIG} ${outputdirectory} )
    set( CMAKE_ARCHIVE_OUTPUT_DIRECTORY_${OUTPUTCONFIG} ${outputdirectory} )
endforeach( OUTPUTCONFIG CMAKE_CONFIGURATION_TYPES )

add_definitions(-DPCL_HELPER_EXPORTS)
set(MySources pcl_helper.cpp;pcl_helper.h;pcl_helper_exports.cpp;pcl_helper_exports.h;pcl_helper_imports.h)
ADD_PRECOMPILED_HEADER("precompiled.h" "precompiled.cpp" MySources)
add_library(pcl_helper SHARED ${MySources})
target_link_libraries(pcl_helper ${PCL_LIBRARIES})
