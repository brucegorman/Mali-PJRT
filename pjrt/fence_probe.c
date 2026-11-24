#include <errno.h>
#include <poll.h>
#include <pthread.h>
#include <stdio.h>
#include <time.h>
#include <unistd.h>

#include "fence_probe.h"
#include "iree/hal/buffer_view_util.h"
#include "iree/modules/hal/module.h"

static double monotonic_ms(void) {
    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);
    return (double)now.tv_sec * 1000.0 + (double)now.tv_nsec / 1000000.0;
}

static iree_status_t create_fence(iree_hal_device_t *device, iree_hal_fence_t **fence) {
    iree_hal_semaphore_t *semaphore = NULL;
    IREE_RETURN_IF_ERROR(iree_hal_semaphore_create(device, IREE_HAL_QUEUE_AFFINITY_ANY, 0,
                                                    IREE_HAL_SEMAPHORE_FLAG_DEFAULT, &semaphore));
    iree_status_t status = iree_hal_fence_create_at(semaphore, 1,
                                                   iree_hal_device_host_allocator(device), fence);
    iree_hal_semaphore_release(semaphore);
    return status;
}

static iree_status_t push_fence(iree_vm_list_t *inputs, iree_hal_fence_t *fence) {
    iree_vm_ref_t ref = iree_hal_fence_retain_ref(fence);
    iree_status_t status = iree_vm_list_push_ref_move(inputs, &ref);
    iree_vm_ref_release(&ref);
    return status;
}

static iree_status_t submit(iree_runtime_session_t *session, iree_string_view_t name,
                           iree_hal_buffer_view_t *input, iree_hal_fence_t *wait,
                           iree_hal_fence_t *signal, iree_hal_buffer_view_t **output) {
    iree_vm_function_t function;
    IREE_RETURN_IF_ERROR(iree_runtime_session_lookup_function(session, name, &function));
    iree_string_view_t model = iree_vm_function_lookup_attr_by_name(&function,
                                                                   IREE_SV("iree.abi.model"));
    if (!iree_string_view_equal(model, IREE_SV("coarse-fences"))) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "Probe requires a coarse-fences export");
    }
    iree_runtime_call_t call;
    IREE_RETURN_IF_ERROR(iree_runtime_call_initialize(session, function, &call));
    iree_vm_list_t *inputs = iree_runtime_call_inputs(&call);
    iree_status_t status = iree_runtime_call_inputs_push_back_buffer_view(&call, input);
    if (iree_status_is_ok(status)) {
        status = push_fence(inputs, wait);
    }
    if (iree_status_is_ok(status)) {
        status = push_fence(inputs, signal);
    }
    if (iree_status_is_ok(status)) {
        status = iree_runtime_call_invoke(&call, 0);
    }
    if (iree_status_is_ok(status)) {
        status = iree_runtime_call_outputs_pop_front_buffer_view(&call, output);
    }
    iree_runtime_call_deinitialize(&call);
    return status;
}

static iree_status_t require_pending(iree_hal_fence_t *fence) {
    iree_status_t status = iree_hal_fence_query(fence);
    if (iree_status_is_deferred(status)) {
        return iree_status_ignore(status);
    }
    return iree_status_is_ok(status)
        ? iree_make_status(IREE_STATUS_FAILED_PRECONDITION, "Output completed before the gate opened")
        : status;
}

struct gate_watchdog {
    iree_hal_fence_t *gate;
    int pipe_read;
    bool timeout;
    double signal_start;
    double signal_ms;
    iree_status_t status;
};

static void *open_gate(void *opaque) {
    struct gate_watchdog *watchdog = opaque;
    struct pollfd trigger = {.fd = watchdog->pipe_read, .events = POLLIN};
    int result;
    do {
        result = poll(&trigger, 1, 1000);
    } while (result < 0 && errno == EINTR);
    watchdog->timeout = result == 0;
    watchdog->signal_start = monotonic_ms();
    watchdog->status = iree_hal_fence_signal(watchdog->gate);
    watchdog->signal_ms = monotonic_ms() - watchdog->signal_start;
    if (result < 0 && iree_status_is_ok(watchdog->status)) {
        watchdog->status = iree_make_status(IREE_STATUS_INTERNAL, "Could not poll the gate trigger");
    }
    return NULL;
}

static iree_status_t check_chain(iree_runtime_session_t *session, iree_hal_device_t *device,
                                bool *blocked) {
    const int32_t values[] = {1, 2, 3, 4};
    const int32_t expected[] = {5, 7, 9, 11};
    const iree_hal_dim_t shape[] = {IREE_ARRAYSIZE(values)};
    iree_hal_buffer_view_t *input = NULL, *middle = NULL, *output = NULL;
    iree_hal_fence_t *gate = NULL, *first_done = NULL, *last_done = NULL;
    bool first_submitted = false, last_submitted = false;
    iree_hal_buffer_params_t params = {
        .type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL, .usage = IREE_HAL_BUFFER_USAGE_DEFAULT,
        .access = IREE_HAL_MEMORY_ACCESS_ALL,
    };
    iree_status_t status = iree_hal_buffer_view_allocate_buffer_copy(
        device, iree_hal_device_allocator(device), IREE_ARRAYSIZE(shape), shape,
        IREE_HAL_ELEMENT_TYPE_SINT_32, IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR,
        params, iree_make_const_byte_span(values, sizeof(values)), &input);
    if (!iree_status_is_ok(status)) goto cleanup;
    status = create_fence(device, &gate);
    if (!iree_status_is_ok(status)) goto cleanup;
    status = create_fence(device, &first_done);
    if (!iree_status_is_ok(status)) goto cleanup;
    status = create_fence(device, &last_done);
    if (!iree_status_is_ok(status)) goto cleanup;

    // A pipe close releases the gate on an asynchronous runtime. The watchdog
    // releases it after one second on a runtime that blocks inside a VM call.
    int trigger[2];
    if (pipe(trigger) != 0) {
        status = iree_make_status(IREE_STATUS_INTERNAL, "Could not create the gate trigger");
        goto cleanup;
    }
    struct gate_watchdog watchdog = {.gate = gate, .pipe_read = trigger[0]};
    pthread_t gate_thread;
    double start = monotonic_ms();
    if (pthread_create(&gate_thread, NULL, open_gate, &watchdog) != 0) {
        close(trigger[0]);
        close(trigger[1]);
        status = iree_make_status(IREE_STATUS_INTERNAL, "Could not start the gate watchdog");
        goto cleanup;
    }
    status = submit(session, IREE_SV("module.double"), input, gate, first_done, &middle);
    double first_returned = monotonic_ms();
    if (iree_status_is_ok(status)) {
        first_submitted = true;
        status = submit(session, IREE_SV("module.add_three"), middle, first_done, last_done, &output);
        last_submitted = iree_status_is_ok(status);
    }
    double last_returned = monotonic_ms();
    iree_status_t pending = iree_ok_status();
    if (last_submitted) {
        pending = require_pending(first_done);
        if (iree_status_is_ok(pending)) {
            pending = require_pending(last_done);
        }
    }
    close(trigger[1]);
    pthread_join(gate_thread, NULL);
    close(trigger[0]);
    *blocked = watchdog.timeout;
    status = iree_status_join(status, watchdog.status);
    if (*blocked) {
        iree_status_ignore(pending);
    } else {
        status = iree_status_join(status, pending);
    }
    if (!iree_status_is_ok(status)) goto cleanup;
    double before_wait = monotonic_ms();
    status = iree_hal_fence_wait(last_done, iree_make_timeout_ms(10000), IREE_HAL_WAIT_FLAG_DEFAULT);
    if (!iree_status_is_ok(status)) goto cleanup;
    double finished = monotonic_ms();
    status = iree_hal_fence_query(first_done);
    if (!iree_status_is_ok(status)) goto cleanup;

    int32_t result[IREE_ARRAYSIZE(expected)];
    if (iree_hal_buffer_view_byte_length(output) != sizeof(result)) {
        status = iree_make_status(IREE_STATUS_INTERNAL, "Fence probe returned an unexpected output size");
        goto cleanup;
    }
    status = iree_hal_device_transfer_d2h(device, iree_hal_buffer_view_buffer(output), 0, result,
                                          sizeof(result), 0, iree_make_timeout_ms(10000));
    if (!iree_status_is_ok(status)) goto cleanup;
    for (size_t i = 0; i < IREE_ARRAYSIZE(expected); ++i) {
        if (result[i] != expected[i]) {
            status = iree_make_status(IREE_STATUS_INTERNAL, "Wrong result at index %zu: %d != %d",
                                      i, result[i], expected[i]);
            goto cleanup;
        }
    }
    if (*blocked && last_returned >= watchdog.signal_start) {
        printf("BLOCKED: %s export was still in submit when the watchdog opened the start fence\n",
               first_returned >= watchdog.signal_start ? "first" : "second");
    } else if (*blocked) {
        puts("BLOCKED: watchdog opened the start fence before the pending checks finished");
    } else {
        puts("PASS: both exports returned while the completion fences were pending");
    }
    printf("PASS: final result=[%d, %d, %d, %d]\n",
           result[0], result[1], result[2], result[3]);
    printf("submit_first_ms=%.3f submit_second_ms=%.3f signal_ms=%.3f final_wait_ms=%.3f\n",
           first_returned - start, last_returned - first_returned,
           watchdog.signal_ms, finished - before_wait);

cleanup:
    // Unblock any accepted work on the error path before releasing the device.
    if (!iree_status_is_ok(status)) {
        iree_hal_fence_t *submitted = last_submitted ? last_done : first_submitted ? first_done : NULL;
        if (submitted != NULL) {
            iree_status_ignore(iree_hal_fence_wait(submitted, iree_make_timeout_ms(1000),
                                                   IREE_HAL_WAIT_FLAG_DEFAULT));
        }
    }
    iree_hal_buffer_view_release(output);
    iree_hal_buffer_view_release(middle);
    iree_hal_buffer_view_release(input);
    iree_hal_fence_release(last_done);
    iree_hal_fence_release(first_done);
    iree_hal_fence_release(gate);
    return status;
}

int run_fence_probe(iree_runtime_instance_t *instance, iree_hal_device_t *device, const char *path) {
    iree_runtime_session_options_t options;
    iree_runtime_session_options_initialize(&options);
    iree_runtime_session_t *session = NULL;
    bool blocked = false;
    iree_status_t status = iree_runtime_session_create_with_device(
        instance, &options, device, iree_allocator_system(), &session);
    if (iree_status_is_ok(status)) {
        status = iree_runtime_session_append_bytecode_module_from_file(session, path);
    }
    if (iree_status_is_ok(status)) {
        status = check_chain(session, device, &blocked);
    }
    iree_runtime_session_release(session);
    if (!iree_status_is_ok(status)) {
        iree_status_fprint(stderr, status);
        iree_status_free(status);
        return 1;
    }
    return blocked ? 3 : 0;
}
