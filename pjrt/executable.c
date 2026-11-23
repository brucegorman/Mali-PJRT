#define _POSIX_C_SOURCE 200809L

#include <fcntl.h>
#include <inttypes.h>
#include <spawn.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

#include "plugin_internal.h"

extern char **environ;

struct PJRT_Executable {
    struct mali_executable_info info;
    int64_t generated_size;
    char fingerprint[17];
    const char *memory_kinds[MALI_MAX_OUTPUTS];
    size_t memory_kind_sizes[MALI_MAX_OUTPUTS];
};

struct PJRT_LoadedExecutable {
    PJRT_Client *client;
    uint64_t handle;
    PJRT_Executable executable;
};

static bool write_program(int fd, const char *data, size_t size) {
    while (size != 0) {
        ssize_t count = write(fd, data, size);
        if (count < 0 && errno == EINTR) {
            continue;
        }
        if (count <= 0) {
            return false;
        }
        data += count;
        size -= (size_t)count;
    }
    return true;
}

static PJRT_Error *compile_module(const PJRT_Program *program, char *output, size_t output_capacity,
                                  int64_t *generated_size) {
    const char *loader = getenv("MALI_GLIBC_LOADER");
    const char *libraries = getenv("MALI_GLIBC_LIBRARIES");
    const char *compiler = getenv("MALI_IREE_COMPILER");
    const char *temporary = getenv("TMPDIR");
    if (loader == NULL || libraries == NULL || compiler == NULL || temporary == NULL) {
        return make_error(PJRT_Error_Code_FAILED_PRECONDITION,
                          "Set the phone IREE compiler environment before calling JAX");
    }
    char input[4096];
    int length = snprintf(input, sizeof(input), "%s/pjrt-mlir-XXXXXX", temporary);
    if (length < 0 || (size_t)length >= sizeof(input) ||
        (size_t)length + sizeof(".vmfb") > output_capacity) {
        return make_error(PJRT_Error_Code_INVALID_ARGUMENT, "Mali compiler temporary path is too long");
    }
    int fd = mkstemp(input);
    if (fd < 0) {
        return make_error(PJRT_Error_Code_INTERNAL, "Could not create a Mali compiler input file");
    }
    snprintf(output, output_capacity, "%s.vmfb", input);
    bool written = write_program(fd, program->code, program->code_size);
    int closed = close(fd);
    PJRT_Error *error = NULL;
    if (!written || closed != 0) {
        error = make_error(PJRT_Error_Code_INTERNAL, "Could not write the Mali compiler input");
    } else {
        char *argv[] = {
            (char *)loader, "--library-path", (char *)libraries, (char *)compiler, input,
            "--iree-input-type=stablehlo", "--iree-hal-target-device=vulkan", "-o", output, NULL,
        };
        pid_t pid;
        if (posix_spawn(&pid, loader, NULL, NULL, argv, environ) != 0) {
            error = make_error(PJRT_Error_Code_UNAVAILABLE, "Could not start IREE on the phone");
        } else {
            int status;
            pid_t waited;
            do {
                waited = waitpid(pid, &status, 0);
            } while (waited < 0 && errno == EINTR);
            if (waited < 0 || !WIFEXITED(status) || WEXITSTATUS(status) != 0) {
                error = make_error(PJRT_Error_Code_INVALID_ARGUMENT,
                                   "IREE could not compile the StableHLO program (see compiler stderr)");
            }
        }
    }
    unlink(input);
    struct stat file;
    if (error == NULL && (stat(output, &file) != 0 || file.st_size <= 0)) {
        error = make_error(PJRT_Error_Code_INTERNAL, "IREE produced no Mali module");
    }
    if (error != NULL) {
        unlink(output);
    } else {
        *generated_size = file.st_size;
    }
    return error;
}

static uint64_t program_fingerprint(uint64_t hash, const char *bytes, size_t length) {
    for (size_t i = 0; i < length; ++i) {
        hash = (hash ^ (unsigned char)bytes[i]) * UINT64_C(1099511628211);
    }
    return hash;
}

static PJRT_Error *client_compile(PJRT_Client_Compile_Args *args) {
    args->executable = NULL;
    if (args->program == NULL || args->program->format_size != 4 ||
        memcmp(args->program->format, "mlir", 4) != 0 || args->program->code_size == 0) {
        return make_error(PJRT_Error_Code_INVALID_ARGUMENT, "Mali requires a StableHLO MLIR program");
    }
    PJRT_LoadedExecutable *loaded = calloc(1, sizeof(*loaded));
    if (loaded == NULL) {
        return make_error(PJRT_Error_Code_RESOURCE_EXHAUSTED, "Could not allocate Mali executable");
    }
    char path[4096];
    PJRT_Error *error = compile_module(args->program, path, sizeof(path),
                                      &loaded->executable.generated_size);
    if (error != NULL) {
        free(loaded);
        return error;
    }
    struct mali_request request = {.operation = MALI_LOAD};
    struct mali_reply reply;
    pthread_mutex_lock(&args->client->worker_lock);
    int status = exchange(args->client, &request, path, strlen(path) + 1,
                          &loaded->executable.info, sizeof(loaded->executable.info), &reply);
    pthread_mutex_unlock(&args->client->worker_lock);
    unlink(path);
    if (status != MALI_OK) {
        free(loaded);
        return transfer_error(status);
    }
    loaded->client = args->client;
    loaded->handle = reply.handle;
    for (uint32_t i = 0; i < loaded->executable.info.outputs; ++i) {
        loaded->executable.memory_kinds[i] = "device";
        loaded->executable.memory_kind_sizes[i] = sizeof("device") - 1;
    }
    uint64_t hash = program_fingerprint(UINT64_C(14695981039346656037),
                                        args->program->code, args->program->code_size);
    hash = program_fingerprint(hash, args->compile_options, args->compile_options_size);
    snprintf(loaded->executable.fingerprint, sizeof(loaded->executable.fingerprint),
             "%016" PRIx64, hash);
    args->executable = loaded;
    return NULL;
}

static PJRT_Error *loaded_execute(PJRT_LoadedExecutable_Execute_Args *args) {
    PJRT_LoadedExecutable *loaded = args->executable;
    PJRT_Client *client = loaded->client;
    size_t outputs = loaded->executable.info.outputs;
    if (args->num_devices != 1 || args->num_args != loaded->executable.info.inputs ||
        (args->execute_device != NULL && args->execute_device != &client->device) ||
        (args->options != NULL && (args->options->num_send_ops || args->options->num_recv_ops))) {
        return make_error(PJRT_Error_Code_INVALID_ARGUMENT, "Unsupported Mali execution configuration");
    }
    uint64_t handles[MALI_TRANSFER_CHUNK / sizeof(uint64_t)];
    struct mali_result results[MALI_MAX_OUTPUTS];
    struct mali_reply reply;
    pthread_mutex_lock(&client->worker_lock);
    if (loaded->handle == 0) {
        pthread_mutex_unlock(&client->worker_lock);
        return make_error(PJRT_Error_Code_FAILED_PRECONDITION, "Mali executable was deleted");
    }
    for (size_t i = 0; i < args->num_args; ++i) {
        PJRT_Buffer *input = args->argument_lists[0][i];
        if (input->client != client || input->handle == 0) {
            pthread_mutex_unlock(&client->worker_lock);
            return make_error(PJRT_Error_Code_INVALID_ARGUMENT, "Invalid Mali execution input");
        }
        handles[i] = input->handle;
    }
    struct mali_request request = {.operation = MALI_EXECUTE, .handle = loaded->handle};
    int status = exchange(client, &request, handles, args->num_args * sizeof(handles[0]),
                          results, outputs * sizeof(results[0]), &reply);
    if (status != MALI_OK) {
        pthread_mutex_unlock(&client->worker_lock);
        return transfer_error(status);
    }
    for (size_t i = 0; i < outputs; ++i) {
        args->output_lists[0][i] = create_buffer(client, &results[i].tensor, results[i].handle);
        if (args->output_lists[0][i] == NULL) {
            for (size_t j = 0; j < outputs; ++j) {
                release_handle(client, results[j].handle);
            }
            for (size_t j = 0; j < i; ++j) {
                free(args->output_lists[0][j]);
                args->output_lists[0][j] = NULL;
            }
            pthread_mutex_unlock(&client->worker_lock);
            return make_error(PJRT_Error_Code_RESOURCE_EXHAUSTED, "Could not record Mali outputs");
        }
    }
    pthread_mutex_unlock(&client->worker_lock);
    if (args->device_complete_events != NULL) {
        args->device_complete_events[0] = buffer_ready();
    }
    return NULL;
}

static void unload(PJRT_LoadedExecutable *loaded) {
    PJRT_Client *client = loaded->client;
    pthread_mutex_lock(&client->worker_lock);
    if (loaded->handle != 0) {
        struct mali_request request = {.operation = MALI_UNLOAD, .handle = loaded->handle};
        struct mali_reply reply;
        exchange(client, &request, NULL, 0, NULL, 0, &reply);
        loaded->handle = 0;
    }
    pthread_mutex_unlock(&client->worker_lock);
}

static PJRT_Error *loaded_destroy(PJRT_LoadedExecutable_Destroy_Args *args) {
    if (args->executable != NULL) {
        unload(args->executable);
        free(args->executable);
    }
    return NULL;
}

static PJRT_Error *loaded_delete(PJRT_LoadedExecutable_Delete_Args *args) {
    unload(args->executable);
    return NULL;
}

static PJRT_Error *loaded_is_deleted(PJRT_LoadedExecutable_IsDeleted_Args *args) {
    pthread_mutex_t *lock = &args->executable->client->worker_lock;
    pthread_mutex_lock(lock);
    args->is_deleted = args->executable->handle == 0;
    pthread_mutex_unlock(lock);
    return NULL;
}

static PJRT_Error *loaded_get_executable(PJRT_LoadedExecutable_GetExecutable_Args *args) {
    args->executable = malloc(sizeof(*args->executable));
    if (args->executable == NULL) {
        return make_error(PJRT_Error_Code_RESOURCE_EXHAUSTED, "Could not copy Mali executable info");
    }
    *args->executable = args->loaded_executable->executable;
    return NULL;
}

static PJRT_Error *loaded_devices(PJRT_LoadedExecutable_AddressableDevices_Args *args) {
    args->addressable_devices = args->executable->client->devices;
    args->num_addressable_devices = 1;
    return NULL;
}

static PJRT_Error *executable_destroy(PJRT_Executable_Destroy_Args *args) {
    free(args->executable);
    return NULL;
}

static PJRT_Error *executable_name(PJRT_Executable_Name_Args *args) {
    args->executable_name = "Mali Vulkan main";
    args->executable_name_size = sizeof("Mali Vulkan main") - 1;
    return NULL;
}

static PJRT_Error *executable_replicas(PJRT_Executable_NumReplicas_Args *args) {
    args->num_replicas = 1;
    return NULL;
}

static PJRT_Error *executable_partitions(PJRT_Executable_NumPartitions_Args *args) {
    args->num_partitions = 1;
    return NULL;
}

static PJRT_Error *executable_outputs(PJRT_Executable_NumOutputs_Args *args) {
    args->num_outputs = args->executable->info.outputs;
    return NULL;
}

static PJRT_Error *executable_memory_kinds(PJRT_Executable_OutputMemoryKinds_Args *args) {
    args->num_outputs = args->executable->info.outputs;
    args->memory_kinds = args->executable->memory_kinds;
    args->memory_kind_sizes = args->executable->memory_kind_sizes;
    return NULL;
}

static PJRT_Error *executable_size(PJRT_Executable_SizeOfGeneratedCodeInBytes_Args *args) {
    args->size_in_bytes = args->executable->generated_size;
    return NULL;
}

static PJRT_Error *executable_fingerprint(PJRT_Executable_Fingerprint_Args *args) {
    args->executable_fingerprint = args->executable->fingerprint;
    args->executable_fingerprint_size = sizeof(args->executable->fingerprint) - 1;
    return NULL;
}

static PJRT_Error *executable_serialize(PJRT_Executable_Serialize_Args *args) {
    (void)args;
    return make_error(PJRT_Error_Code_UNIMPLEMENTED, "Mali executable serialization is unavailable");
}

static PJRT_Error *executable_deserialize(PJRT_Executable_DeserializeAndLoad_Args *args) {
    (void)args;
    return make_error(PJRT_Error_Code_UNIMPLEMENTED, "Mali executable deserialization is unavailable");
}

static PJRT_Error *executable_optimized_program(PJRT_Executable_OptimizedProgram_Args *args) {
    (void)args;
    return make_error(PJRT_Error_Code_UNIMPLEMENTED, "Mali executable has no optimized XLA HLO");
}

static PJRT_Error *executable_cost_analysis(PJRT_Executable_GetCostAnalysis_Args *args) {
    (void)args;
    return make_error(PJRT_Error_Code_UNIMPLEMENTED, "Mali executable cost analysis is unavailable");
}

static PJRT_Error *executable_memory_stats(PJRT_Executable_GetCompiledMemoryStats_Args *args) {
    (void)args;
    return make_error(PJRT_Error_Code_UNIMPLEMENTED, "Mali executable memory statistics are unavailable");
}

void register_executable_api(PJRT_Api *api) {
    api->PJRT_Client_Compile = client_compile;
    api->PJRT_LoadedExecutable_Execute = loaded_execute;
    api->PJRT_LoadedExecutable_Destroy = loaded_destroy;
    api->PJRT_LoadedExecutable_Delete = loaded_delete;
    api->PJRT_LoadedExecutable_IsDeleted = loaded_is_deleted;
    api->PJRT_LoadedExecutable_GetExecutable = loaded_get_executable;
    api->PJRT_LoadedExecutable_AddressableDevices = loaded_devices;
    api->PJRT_Executable_Destroy = executable_destroy;
    api->PJRT_Executable_Name = executable_name;
    api->PJRT_Executable_NumReplicas = executable_replicas;
    api->PJRT_Executable_NumPartitions = executable_partitions;
    api->PJRT_Executable_NumOutputs = executable_outputs;
    api->PJRT_Executable_OutputMemoryKinds = executable_memory_kinds;
    api->PJRT_Executable_SizeOfGeneratedCodeInBytes = executable_size;
    api->PJRT_Executable_Fingerprint = executable_fingerprint;
    api->PJRT_Executable_Serialize = executable_serialize;
    api->PJRT_Executable_DeserializeAndLoad = executable_deserialize;
    api->PJRT_Executable_OptimizedProgram = executable_optimized_program;
    api->PJRT_Executable_GetCostAnalysis = executable_cost_analysis;
    api->PJRT_Executable_GetCompiledMemoryStats = executable_memory_stats;
}
