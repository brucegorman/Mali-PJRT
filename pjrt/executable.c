#define _POSIX_C_SOURCE 200809L

#include <fcntl.h>
#include <inttypes.h>
#include <spawn.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/wait.h>
#include <unistd.h>

#include "executable_image.h"

extern char **environ;

struct PJRT_Executable {
    uint32_t inputs;
    uint32_t outputs;
    struct executable_image *image;
    char fingerprint[17];
    const char *memory_kinds[MALI_MAX_OUTPUTS];
    size_t memory_kind_sizes[MALI_MAX_OUTPUTS];
    PJRT_Buffer_Type output_types[MALI_MAX_OUTPUTS];
    size_t output_ranks[MALI_MAX_OUTPUTS];
    int64_t output_dims[MALI_MAX_OUTPUTS * MALI_MAX_RANK];
};

struct PJRT_LoadedExecutable {
    PJRT_Client *client;
    struct mali_resource *resource;
    size_t output_sizes[MALI_MAX_OUTPUTS];
    PJRT_Executable executable;
};

struct PJRT_SerializedExecutable {
    struct executable_image *image;
};

static bool write_file(int fd, const char *data, size_t size) {
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

static PJRT_Error *compile_module(const PJRT_Program *program, const char *target,
                                  char *output, size_t output_capacity) {
    const char *loader = getenv("MALI_GLIBC_LOADER");
    const char *libraries = getenv("MALI_GLIBC_LIBRARIES");
    const char *compiler = getenv("MALI_IREE_COMPILER");
    const char *temporary = getenv("TMPDIR");
    if (loader == NULL || libraries == NULL || compiler == NULL || temporary == NULL) {
        return make_error(PJRT_Error_Code_FAILED_PRECONDITION,
                          "Set the phone IREE compiler environment before calling JAX");
    }
    const char *spec = getenv("MALI_IREE_TUNING_SPEC");
    char tuning_flag[4096];
    char *tuning = NULL;
    if (spec != NULL && *spec != '\0') {
        if (target == NULL) {
            return make_error(PJRT_Error_Code_FAILED_PRECONDITION,
                              "MALI_IREE_TUNING_SPEC requires a Mali-G57 device");
        }
        int size = snprintf(tuning_flag, sizeof(tuning_flag), "--iree-codegen-transform-dialect-library=%s", spec);
        if (size < 0 || (size_t)size >= sizeof(tuning_flag)) {
            return make_error(PJRT_Error_Code_INVALID_ARGUMENT, "Mali tuning spec path is too long");
        }
        tuning = tuning_flag;
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
    bool written = write_file(fd, program->code, program->code_size);
    int closed = close(fd);
    PJRT_Error *error = NULL;
    if (!written || closed != 0) {
        error = make_error(PJRT_Error_Code_INTERNAL, "Could not write the Mali compiler input");
    } else {
        char *argv[] = {
            (char *)loader, "--library-path", (char *)libraries, (char *)compiler, input,
            "--iree-input-type=stablehlo", "--iree-hal-target-device=vulkan", "-o", output,
            (char *)target, tuning, NULL,
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
    if (error != NULL) {
        unlink(output);
    }
    return error;
}

static PJRT_Error *load_image(PJRT_Client *client, const char *path, struct executable_image *image,
                              PJRT_LoadedExecutable **out) {
    PJRT_LoadedExecutable *loaded = calloc(1, sizeof(*loaded));
    if (loaded == NULL) {
        return make_error(PJRT_Error_Code_RESOURCE_EXHAUSTED, "Could not allocate Mali executable");
    }
    loaded->resource = create_resource(client, MALI_UNLOAD, 0, ready_event());
    if (loaded->resource == NULL) {
        free(loaded);
        return make_error(PJRT_Error_Code_RESOURCE_EXHAUSTED, "Could not allocate Mali executable handle");
    }
    struct mali_request request = {.operation = MALI_LOAD};
    struct mali_reply reply;
    struct mali_executable_info info;
    pthread_mutex_lock(&client->worker_lock);
    int status = exchange(client, &request, path, strlen(path) + 1,
                          &info, sizeof(info), &reply);
    pthread_mutex_unlock(&client->worker_lock);
    if (status != MALI_OK) {
        release_resource(loaded->resource);
        free(loaded);
        return transfer_error(MALI_LOAD, status);
    }
    loaded->client = client;
    loaded->resource->handle = reply.handle;
    if (reply.handle == 0 || info.inputs > MALI_TRANSFER_CHUNK / sizeof(uint64_t) ||
        info.outputs > MALI_MAX_OUTPUTS) {
        release_resource(loaded->resource);
        free(loaded);
        return make_error(PJRT_Error_Code_INTERNAL, "Invalid Mali executable metadata");
    }
    loaded->executable.inputs = info.inputs;
    loaded->executable.outputs = info.outputs;
    size_t dim_offset = 0;
    for (uint32_t i = 0; i < info.outputs; ++i) {
        const struct mali_tensor *tensor = &info.output_tensors[i];
        if (!tensor_size(tensor, &loaded->output_sizes[i])) {
            release_resource(loaded->resource);
            free(loaded);
            return make_error(PJRT_Error_Code_UNIMPLEMENTED, "Unsupported Mali output storage size");
        }
        loaded->executable.memory_kinds[i] = "device";
        loaded->executable.memory_kind_sizes[i] = sizeof("device") - 1;
        loaded->executable.output_types[i] = tensor->type;
        loaded->executable.output_ranks[i] = tensor->rank;
        memcpy(loaded->executable.output_dims + dim_offset, tensor->dims,
               tensor->rank * sizeof(tensor->dims[0]));
        dim_offset += tensor->rank;
    }
    loaded->executable.image = image;
    retain_image(image);
    snprintf(loaded->executable.fingerprint, sizeof(loaded->executable.fingerprint),
             "%016" PRIx64, image_fingerprint(image));
    *out = loaded;
    return NULL;
}

static PJRT_Error *client_compile(PJRT_Client_Compile_Args *args) {
    args->executable = NULL;
    if (args->program == NULL || args->program->format_size != 4 ||
        memcmp(args->program->format, "mlir", 4) != 0 || args->program->code_size == 0) {
        return make_error(PJRT_Error_Code_INVALID_ARGUMENT, "Mali requires a StableHLO MLIR program");
    }
    PJRT_Error *error = validate_compile_options(
        args->client, args->compile_options, args->compile_options_size);
    if (error != NULL) {
        return error;
    }
    char path[4096];
    const struct mali_device_info *device = &args->client->device.description.info;
    const char *target = strncmp(device->name, "Mali-G57", sizeof("Mali-G57") - 1) == 0
        ? "--iree-vulkan-target=mali-g57" : NULL;
    error = compile_module(args->program, target, path, sizeof(path));
    if (error != NULL) {
        return error;
    }
    struct executable_image *image;
    error = read_compiled_image(path, device, &image);
    if (error == NULL) {
        error = load_image(args->client, path, image, &args->executable);
        release_image(image);
    }
    unlink(path);
    return error;
}

static PJRT_Error *loaded_execute(PJRT_LoadedExecutable_Execute_Args *args) {
    PJRT_LoadedExecutable *loaded = args->executable;
    PJRT_Client *client = loaded->client;
    size_t outputs = loaded->executable.outputs;
    if (args->num_devices != 1 || args->num_args != loaded->executable.inputs ||
        (args->execute_device != NULL && args->execute_device != &client->device) ||
        (args->options != NULL && (args->options->num_send_ops || args->options->num_recv_ops))) {
        return make_error(PJRT_Error_Code_INVALID_ARGUMENT, "Unsupported Mali execution configuration");
    }
    size_t inputs = args->num_args;
    struct mali_execution *job = calloc(1, sizeof(*job) + (inputs + outputs) * sizeof(job->buffers[0]));
    PJRT_Event *done = create_event();
    PJRT_Buffer *out[MALI_MAX_OUTPUTS];
    size_t created = 0;
    PJRT_Error *error = NULL;
    if (job == NULL || done == NULL) {
        error = make_error(PJRT_Error_Code_RESOURCE_EXHAUSTED, "Could not allocate Mali execution");
        goto cleanup;
    }
    job->done = done;
    job->inputs = inputs;
    job->outputs = outputs;
    size_t dim_offset = 0;
    for (; created < outputs; ++created) {
        struct mali_tensor tensor = {
            .size = loaded->output_sizes[created],
            .type = loaded->executable.output_types[created],
            .rank = loaded->executable.output_ranks[created],
        };
        memcpy(tensor.dims, loaded->executable.output_dims + dim_offset,
               tensor.rank * sizeof(tensor.dims[0]));
        dim_offset += tensor.rank;
        out[created] = create_buffer(client, &tensor, done);
        if (out[created] == NULL) {
            error = make_error(PJRT_Error_Code_RESOURCE_EXHAUSTED, "Could not record Mali outputs");
            goto cleanup;
        }
        job->buffers[inputs + created] = out[created]->resource;
        job->output_sizes[created] = tensor.size;
    }

    pthread_mutex_lock(&client->dispatch_lock);
    if (loaded->resource == NULL || client->stop_dispatch) {
        pthread_mutex_unlock(&client->dispatch_lock);
        error = make_error(PJRT_Error_Code_FAILED_PRECONDITION, "Mali executable or client was closed");
        goto cleanup;
    }
    for (size_t i = 0; i < inputs; ++i) {
        PJRT_Buffer *input = args->argument_lists[0][i];
        if (input == NULL || input->client != client || input->resource == NULL) {
            pthread_mutex_unlock(&client->dispatch_lock);
            error = make_error(PJRT_Error_Code_INVALID_ARGUMENT, "Invalid Mali execution input");
            goto cleanup;
        }
        job->buffers[i] = input->resource;
    }
    job->executable = loaded->resource;
    retain_resource(job->executable);
    for (size_t i = 0; i < inputs + outputs; ++i) {
        retain_resource(job->buffers[i]);
    }
    for (size_t i = 0; i < outputs; ++i) {
        args->output_lists[0][i] = out[i];
    }
    if (args->device_complete_events != NULL) {
        retain_event(done);
        args->device_complete_events[0] = done;
    }
    enqueue_execution(client, job);
    pthread_mutex_unlock(&client->dispatch_lock);
    return NULL;

cleanup:
    for (size_t i = 0; i < created; ++i) {
        release_resource(out[i]->resource);
        free(out[i]);
    }
    release_event(done);
    free(job);
    return error;
}

static PJRT_Error *loaded_destroy(PJRT_LoadedExecutable_Destroy_Args *args) {
    if (args->executable != NULL) {
        delete_resource(args->executable->client, &args->executable->resource);
        release_image(args->executable->executable.image);
        free(args->executable);
    }
    return NULL;
}

static PJRT_Error *loaded_delete(PJRT_LoadedExecutable_Delete_Args *args) {
    delete_resource(args->executable->client, &args->executable->resource);
    return NULL;
}

static PJRT_Error *loaded_is_deleted(PJRT_LoadedExecutable_IsDeleted_Args *args) {
    args->is_deleted = resource_deleted(args->executable->client, &args->executable->resource);
    return NULL;
}

static PJRT_Error *loaded_get_executable(PJRT_LoadedExecutable_GetExecutable_Args *args) {
    args->executable = malloc(sizeof(*args->executable));
    if (args->executable == NULL) {
        return make_error(PJRT_Error_Code_RESOURCE_EXHAUSTED, "Could not copy Mali executable info");
    }
    *args->executable = args->loaded_executable->executable;
    retain_image(args->executable->image);
    return NULL;
}

static PJRT_Error *loaded_devices(PJRT_LoadedExecutable_AddressableDevices_Args *args) {
    args->addressable_devices = args->executable->client->devices;
    args->num_addressable_devices = 1;
    return NULL;
}

static PJRT_Error *executable_destroy(PJRT_Executable_Destroy_Args *args) {
    if (args->executable != NULL) {
        release_image(args->executable->image);
        free(args->executable);
    }
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
    args->num_outputs = args->executable->outputs;
    return NULL;
}

static PJRT_Error *executable_memory_kinds(PJRT_Executable_OutputMemoryKinds_Args *args) {
    args->num_outputs = args->executable->outputs;
    args->memory_kinds = args->executable->memory_kinds;
    args->memory_kind_sizes = args->executable->memory_kind_sizes;
    return NULL;
}

static PJRT_Error *executable_output_types(PJRT_Executable_OutputElementTypes_Args *args) {
    args->output_types = args->executable->output_types;
    args->num_output_types = args->executable->outputs;
    return NULL;
}

static PJRT_Error *executable_output_dimensions(PJRT_Executable_OutputDimensions_Args *args) {
    args->num_outputs = args->executable->outputs;
    args->dim_sizes = args->executable->output_ranks;
    args->dims = args->executable->output_dims;
    return NULL;
}

static PJRT_Error *executable_size(PJRT_Executable_SizeOfGeneratedCodeInBytes_Args *args) {
    args->size_in_bytes = (int64_t)image_vmfb_size(args->executable->image);
    return NULL;
}

static PJRT_Error *executable_fingerprint(PJRT_Executable_Fingerprint_Args *args) {
    args->executable_fingerprint = args->executable->fingerprint;
    args->executable_fingerprint_size = sizeof(args->executable->fingerprint) - 1;
    return NULL;
}

static void serialized_destroy(PJRT_SerializedExecutable *serialized) {
    release_image(serialized->image);
    free(serialized);
}

static PJRT_Error *executable_serialize(PJRT_Executable_Serialize_Args *args) {
    PJRT_SerializedExecutable *serialized = malloc(sizeof(*serialized));
    if (serialized == NULL) {
        return make_error(PJRT_Error_Code_RESOURCE_EXHAUSTED, "Could not serialize the Mali executable");
    }
    serialized->image = args->executable->image;
    retain_image(serialized->image);
    args->serialized_bytes = image_serialized(serialized->image, &args->serialized_bytes_size);
    args->serialized_executable = serialized;
    args->serialized_executable_deleter = serialized_destroy;
    return NULL;
}

static PJRT_Error *executable_deserialize(PJRT_Executable_DeserializeAndLoad_Args *args) {
    args->loaded_executable = NULL;
    struct executable_image *image;
    PJRT_Error *error = read_serialized_image(
        args->serialized_executable, args->serialized_executable_size,
        &args->client->device.description.info, &image);
    if (error != NULL) {
        return error;
    }
    const char *temporary = getenv("TMPDIR");
    char path[4096];
    int length = temporary != NULL
        ? snprintf(path, sizeof(path), "%s/pjrt-vmfb-XXXXXX", temporary) : -1;
    int fd = length >= 0 && (size_t)length < sizeof(path) ? mkstemp(path) : -1;
    if (fd < 0) {
        release_image(image);
        return make_error(PJRT_Error_Code_INTERNAL, "Could not stage the cached Mali module");
    }
    bool written = write_file(fd, image_vmfb(image), image_vmfb_size(image));
    int closed = close(fd);
    if (written && closed == 0) {
        error = load_image(args->client, path, image, &args->loaded_executable);
    } else {
        error = make_error(PJRT_Error_Code_INTERNAL, "Could not write the cached Mali module");
    }
    unlink(path);
    release_image(image);
    return error;
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
    api->PJRT_Executable_OutputElementTypes = executable_output_types;
    api->PJRT_Executable_OutputDimensions = executable_output_dimensions;
    api->PJRT_Executable_SizeOfGeneratedCodeInBytes = executable_size;
    api->PJRT_Executable_Fingerprint = executable_fingerprint;
    api->PJRT_Executable_Serialize = executable_serialize;
    api->PJRT_Executable_DeserializeAndLoad = executable_deserialize;
    api->PJRT_Executable_OptimizedProgram = executable_optimized_program;
    api->PJRT_Executable_GetCostAnalysis = executable_cost_analysis;
    api->PJRT_Executable_GetCompiledMemoryStats = executable_memory_stats;
}
