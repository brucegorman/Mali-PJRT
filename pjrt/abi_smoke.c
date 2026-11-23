#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "pjrt_c_api.h"

static const PJRT_Api *api;

static void expect(bool condition, const char *message) {
    if (!condition) {
        fprintf(stderr, "%s\n", message);
        exit(1);
    }
}

static void check_error(PJRT_Error *error, PJRT_Error_Code expected_code, const char *expected) {
    if (error != NULL) {
        PJRT_Error_Message_Args message = {
            .struct_size = PJRT_Error_Message_Args_STRUCT_SIZE, .error = error,
        };
        PJRT_Error_GetCode_Args code = {
            .struct_size = PJRT_Error_GetCode_Args_STRUCT_SIZE, .error = error,
        };
        api->PJRT_Error_Message(&message);
        expect(api->PJRT_Error_GetCode(&code) == NULL, "Could not read PJRT error code");
        if (expected == NULL || code.code != expected_code ||
            strstr(message.message, expected) == NULL) {
            fprintf(stderr, "Unexpected PJRT error: %.*s\n", (int)message.message_size,
                    message.message);
            exit(1);
        }
        PJRT_Error_Destroy_Args destroy = {
            .struct_size = PJRT_Error_Destroy_Args_STRUCT_SIZE, .error = error,
        };
        api->PJRT_Error_Destroy(&destroy);
    } else {
        expect(expected == NULL, "Compile options should have been rejected");
    }
}

int main(int argc, char **argv) {
    expect(argc == 2, "Usage: abi_smoke LIBPJRT_PATH");
    void *library = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
    expect(library != NULL, library == NULL ? dlerror() : "");
    const PJRT_Api *(*get_api)(void) = (const PJRT_Api *(*)(void))dlsym(library, "GetPjrtApi");
    expect(get_api != NULL, "GetPjrtApi is missing");
    api = get_api();
    PJRT_Plugin_Initialize_Args initialization = {.struct_size = PJRT_Plugin_Initialize_Args_STRUCT_SIZE};
    check_error(api->PJRT_Plugin_Initialize(&initialization), 0, NULL);
    PJRT_Client_Create_Args create = {.struct_size = PJRT_Client_Create_Args_STRUCT_SIZE};
    check_error(api->PJRT_Client_Create(&create), 0, NULL);

    char mlir[] =
        "module { func.func public @main(%x: tensor<2x3xui32>, %y: tensor<f32>)"
        " -> (tensor<2x3xui32>, tensor<f32>) { return %x, %y : tensor<2x3xui32>, tensor<f32> } }";
    PJRT_Program program = {
        .struct_size = PJRT_Program_STRUCT_SIZE, .code = mlir, .code_size = sizeof(mlir) - 1,
        .format = "mlir", .format_size = 4,
    };
    PJRT_Client_Compile_Args compile = {
        .struct_size = PJRT_Client_Compile_Args_STRUCT_SIZE, .client = create.client,
        .program = &program,
    };
    const char replicas[] = {0x1a, 0x02, 0x20, 0x02};       // build.num_replicas = 2
    const char partitions[] = {0x1a, 0x02, 0x28, 0x02};     // build.num_partitions = 2
    const char assignment[] = {0x1a, 0x0b, 0x4a, 0x09, 0x08, 1, 0x10, 1, 0x1a, 3,
                               0x0a, 1, 1};              // build.device_assignment = [[1]]
    const char malformed[] = {0x1a, 0x02, 0x20};            // Truncated build options.
    const char debug[] = {0x1a, 0x05, 0x1a, 0x03, 0xf8, 0x01, 0x01};
    const struct {
        const char *bytes;
        size_t size;
        PJRT_Error_Code code;
        const char *message;
    } invalid[] = {
        {replicas, sizeof(replicas), PJRT_Error_Code_INVALID_ARGUMENT, "one replica"},
        {partitions, sizeof(partitions), PJRT_Error_Code_INVALID_ARGUMENT, "one partition"},
        {assignment, sizeof(assignment), PJRT_Error_Code_INVALID_ARGUMENT, "device 0"},
        {malformed, sizeof(malformed), PJRT_Error_Code_INVALID_ARGUMENT, "Malformed XLA compile options"},
        {debug, sizeof(debug), PJRT_Error_Code_INVALID_ARGUMENT, "custom XLA debug options"},
    };
    for (size_t i = 0; i < sizeof(invalid) / sizeof(invalid[0]); ++i) {
        compile.compile_options = invalid[i].bytes;
        compile.compile_options_size = invalid[i].size;
        check_error(api->PJRT_Client_Compile(&compile), invalid[i].code, invalid[i].message);
        expect(compile.executable == NULL, "Invalid compile options produced an executable");
    }
    puts("PASS: PJRT C API rejected invalid and unsupported compile options");

    compile.compile_options = NULL;
    compile.compile_options_size = 0;
    check_error(api->PJRT_Client_Compile(&compile), 0, NULL);
    PJRT_LoadedExecutable_GetExecutable_Args describe = {
        .struct_size = PJRT_LoadedExecutable_GetExecutable_Args_STRUCT_SIZE,
        .loaded_executable = compile.executable,
    };
    check_error(api->PJRT_LoadedExecutable_GetExecutable(&describe), 0, NULL);
    PJRT_Executable_OutputElementTypes_Args types = {
        .struct_size = PJRT_Executable_OutputElementTypes_Args_STRUCT_SIZE,
        .executable = describe.executable,
    };
    PJRT_Executable_OutputDimensions_Args dims = {
        .struct_size = PJRT_Executable_OutputDimensions_Args_STRUCT_SIZE,
        .executable = describe.executable,
    };
    check_error(api->PJRT_Executable_OutputElementTypes(&types), 0, NULL);
    check_error(api->PJRT_Executable_OutputDimensions(&dims), 0, NULL);
    expect(types.num_output_types == 2 && types.output_types[0] == PJRT_Buffer_Type_U32 &&
           types.output_types[1] == PJRT_Buffer_Type_F32, "Wrong output dtypes before execution");
    expect(dims.num_outputs == 2 && dims.dim_sizes[0] == 2 && dims.dim_sizes[1] == 0 &&
           dims.dims[0] == 2 && dims.dims[1] == 3, "Wrong output shapes before execution");
    puts("PASS: PJRT C API reported uint32[2,3] and scalar float32 before execution");

    PJRT_Executable_Destroy_Args destroy_executable = {
        .struct_size = PJRT_Executable_Destroy_Args_STRUCT_SIZE, .executable = describe.executable,
    };
    PJRT_LoadedExecutable_Destroy_Args destroy_loaded = {
        .struct_size = PJRT_LoadedExecutable_Destroy_Args_STRUCT_SIZE, .executable = compile.executable,
    };
    PJRT_Client_Destroy_Args destroy_client = {
        .struct_size = PJRT_Client_Destroy_Args_STRUCT_SIZE, .client = create.client,
    };
    check_error(api->PJRT_Executable_Destroy(&destroy_executable), 0, NULL);
    check_error(api->PJRT_LoadedExecutable_Destroy(&destroy_loaded), 0, NULL);
    check_error(api->PJRT_Client_Destroy(&destroy_client), 0, NULL);
    dlclose(library);
    return 0;
}
