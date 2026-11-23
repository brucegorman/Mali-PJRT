#define _POSIX_C_SOURCE 200809L

#include <dlfcn.h>
#include <semaphore.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

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

static PJRT_Buffer *upload(PJRT_Client *client, const void *data, PJRT_Buffer_Type type,
                           const int64_t *dims, size_t rank) {
    PJRT_Client_BufferFromHostBuffer_Args args = {
        .struct_size = PJRT_Client_BufferFromHostBuffer_Args_STRUCT_SIZE,
        .client = client, .data = data, .type = type, .dims = dims, .num_dims = rank,
        .host_buffer_semantics = PJRT_HostBufferSemantics_kImmutableOnlyDuringCall,
    };
    check_error(api->PJRT_Client_BufferFromHostBuffer(&args), 0, NULL);
    PJRT_Event_Destroy_Args done = {
        .struct_size = PJRT_Event_Destroy_Args_STRUCT_SIZE, .event = args.done_with_host_buffer,
    };
    check_error(api->PJRT_Event_Destroy(&done), 0, NULL);
    return args.buffer;
}

static void destroy_buffer(PJRT_Buffer *buffer) {
    PJRT_Buffer_Destroy_Args args = {
        .struct_size = PJRT_Buffer_Destroy_Args_STRUCT_SIZE, .buffer = buffer,
    };
    check_error(api->PJRT_Buffer_Destroy(&args), 0, NULL);
}

static void destroy_event(PJRT_Event *event) {
    PJRT_Event_Destroy_Args args = {
        .struct_size = PJRT_Event_Destroy_Args_STRUCT_SIZE, .event = event,
    };
    check_error(api->PJRT_Event_Destroy(&args), 0, NULL);
}

static void await(PJRT_Event *event, PJRT_Error_Code code, const char *message) {
    PJRT_Event_Await_Args args = {
        .struct_size = PJRT_Event_Await_Args_STRUCT_SIZE, .event = event,
    };
    check_error(api->PJRT_Event_Await(&args), code, message);
}

static void expect_pending(PJRT_Event *event) {
    PJRT_Event_IsReady_Args args = {
        .struct_size = PJRT_Event_IsReady_Args_STRUCT_SIZE, .event = event,
    };
    check_error(api->PJRT_Event_IsReady(&args), 0, NULL);
    expect(!args.is_ready, "Mali event was ready before the worker resumed");
}

static PJRT_Event *buffer_event(PJRT_Buffer *buffer) {
    PJRT_Buffer_ReadyEvent_Args args = {
        .struct_size = PJRT_Buffer_ReadyEvent_Args_STRUCT_SIZE, .buffer = buffer,
    };
    check_error(api->PJRT_Buffer_ReadyEvent(&args), 0, NULL);
    return args.event;
}

static PJRT_Event *dispatch(PJRT_LoadedExecutable *executable, PJRT_Buffer **inputs,
                            PJRT_Buffer **outputs) {
    PJRT_Buffer *const *argument_lists[] = {inputs};
    PJRT_Buffer **output_lists[] = {outputs};
    PJRT_Event *done = NULL;
    PJRT_LoadedExecutable_Execute_Args args = {
        .struct_size = PJRT_LoadedExecutable_Execute_Args_STRUCT_SIZE, .executable = executable,
        .argument_lists = argument_lists, .num_devices = 1, .num_args = 2,
        .output_lists = output_lists, .device_complete_events = &done,
    };
    check_error(api->PJRT_LoadedExecutable_Execute(&args), 0, NULL);
    expect(done != NULL, "Execute did not return a completion event");
    return done;
}

struct callback_test {
    PJRT_Event *dependent;
    sem_t ran;
};

static void on_first_ready(PJRT_Error *error, void *opaque) {
    struct callback_test *test = opaque;
    check_error(error, 0, NULL);
    // This re-entry must not block dispatch of the second job.
    await(test->dependent, 0, NULL);
    expect(sem_post(&test->ran) == 0, "Could not signal the completion callback");
}

static void pause_worker(pid_t worker) {
    expect(kill(worker, SIGSTOP) == 0, "Could not stop the test worker");
    char path[64];
    snprintf(path, sizeof(path), "/proc/%ld/status", (long)worker);
    for (;;) {
        FILE *status = fopen(path, "r");
        expect(status != NULL, "Could not read the test worker state");
        char line[256];
        bool stopped = false;
        while (fgets(line, sizeof(line), status) != NULL) {
            if (strncmp(line, "State:", 6) == 0) {
                stopped = line[6 + strspn(line + 6, " \t")] == 'T';
                break;
            }
        }
        fclose(status);
        if (stopped) {
            return;
        }
        nanosleep(&(struct timespec){.tv_nsec = 1000000}, NULL);
    }
}

static void check_async(PJRT_Client *client, PJRT_LoadedExecutable *executable, pid_t worker) {
    const uint32_t values[] = {0, 1, 2, 3, 4, UINT32_MAX};
    const float scalar = 2.5f;
    PJRT_Buffer *source[] = {
        upload(client, values, PJRT_Buffer_Type_U32, (const int64_t[]){2, 3}, 2),
        upload(client, &scalar, PJRT_Buffer_Type_F32, NULL, 0),
    };
    alarm(15);
    pause_worker(worker);
    PJRT_Buffer *first[2], *second[2];
    PJRT_Event *first_done = dispatch(executable, source, first);
    PJRT_Event *second_done = dispatch(executable, first, second);
    expect_pending(first_done);
    expect_pending(second_done);
    PJRT_Event *first_buffer_ready = buffer_event(first[0]);
    expect_pending(first_buffer_ready);
    PJRT_Buffer_OnDeviceSizeInBytes_Args size = {
        .struct_size = PJRT_Buffer_OnDeviceSizeInBytes_Args_STRUCT_SIZE, .buffer = second[0],
    };
    check_error(api->PJRT_Buffer_OnDeviceSizeInBytes(&size), 0, NULL);
    expect(size.on_device_size_in_bytes == sizeof(values), "Wrong pending output size");

    struct callback_test callback = {.dependent = second_done};
    expect(sem_init(&callback.ran, 0, 0) == 0, "Could not initialize callback semaphore");
    PJRT_Event_OnReady_Args notify = {
        .struct_size = PJRT_Event_OnReady_Args_STRUCT_SIZE, .event = first_done,
        .callback = on_first_ready, .user_arg = &callback,
    };
    check_error(api->PJRT_Event_OnReady(&notify), 0, NULL);
    destroy_event(first_done);

    PJRT_Buffer_Delete_Args delete = {
        .struct_size = PJRT_Buffer_Delete_Args_STRUCT_SIZE, .buffer = first[0],
    };
    check_error(api->PJRT_Buffer_Delete(&delete), 0, NULL);
    PJRT_Event *deleted = buffer_event(first[0]);
    await(deleted, PJRT_Error_Code_FAILED_PRECONDITION, "Mali buffer was deleted");
    destroy_event(deleted);
    expect_pending(first_buffer_ready);
    for (size_t i = 0; i < 2; ++i) {
        destroy_buffer(source[i]);
        destroy_buffer(first[i]);
    }
    PJRT_LoadedExecutable_Destroy_Args unload = {
        .struct_size = PJRT_LoadedExecutable_Destroy_Args_STRUCT_SIZE, .executable = executable,
    };
    check_error(api->PJRT_LoadedExecutable_Destroy(&unload), 0, NULL);
    expect(kill(worker, SIGCONT) == 0, "Could not resume the test worker");

    await(second_done, 0, NULL);
    await(first_buffer_ready, 0, NULL);
    expect(sem_wait(&callback.ran) == 0, "Completion callback did not run");
    sem_destroy(&callback.ran);
    destroy_event(first_buffer_ready);
    destroy_event(second_done);

    uint32_t result[6];
    float scalar_result;
    PJRT_Buffer_ToHostBuffer_Args read = {
        .struct_size = PJRT_Buffer_ToHostBuffer_Args_STRUCT_SIZE,
        .src = second[0], .dst = result, .dst_size = sizeof(result),
    };
    check_error(api->PJRT_Buffer_ToHostBuffer(&read), 0, NULL);
    await(read.event, 0, NULL);
    destroy_event(read.event);
    read.src = second[1];
    read.dst = &scalar_result;
    read.dst_size = sizeof(scalar_result);
    check_error(api->PJRT_Buffer_ToHostBuffer(&read), 0, NULL);
    await(read.event, 0, NULL);
    destroy_event(read.event);
    expect(memcmp(values, result, sizeof(values)) == 0 && scalar_result == scalar,
           "Queued results were wrong after deleting their inputs and executable");
    destroy_buffer(second[0]);
    destroy_buffer(second[1]);
    alarm(0);
    puts("PASS: pending events, two queued executions, early deletion, and reentrant callback");
}

int main(int argc, char **argv) {
    expect(argc == 2 || argc == 3, "Usage: abi_smoke LIBPJRT_PATH [WORKER_PID]");
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
    if (argc == 3) {
        long worker = strtol(argv[2], NULL, 10);
        expect(worker > 1 && worker <= INT32_MAX, "Invalid test worker PID");
        check_async(create.client, compile.executable, (pid_t)worker);
        compile.executable = NULL;
    }

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
