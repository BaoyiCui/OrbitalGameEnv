cc_library(
    name = "OrbitLib_x86",
    srcs = ["so/X86/libOrbit.so"],
    hdrs = ["so/OrbitHPOP.h"],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "OrbitLib_arm",
    srcs = ["so/ARM/libOrbit.so"],
    hdrs = ["so/OrbitHPOP.h"],
    visibility = ["//visibility:public"],
)

# cc_library(
#     name = "OrbitLibPy_x86",
#     srcs = ["src/OrbitHPOPPy.cpp"],
#     hdrs = ["include/OrbitHPOPPy.h"],
#     linkopts = [
#         "-lboost_python311",
#     ],
#     linkshared = 1,
#     visibility = ["//visibility:public"],
#     deps = [
#         ":OrbitLib_x86",
#     ],
# )
