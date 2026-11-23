#include <stdlib.h>

#include "plugin_internal.h"

struct mali_callback {
    PJRT_Event_OnReadyCallback function;
    void *argument;
    int status;
    struct mali_callback *next;
};

struct PJRT_Event {
    bool permanent;
    bool deleted_buffer;
    pthread_mutex_t lock;
    pthread_cond_t completed;
    atomic_size_t references;
    bool ready;
    int status;
    struct mali_callback *callback_head;
    struct mali_callback *callback_tail;
};

PJRT_Event *ready_event(void) {
    static PJRT_Event event = {.permanent = true, .ready = true};
    return &event;
}

PJRT_Event *deleted_event(void) {
    static PJRT_Event event = {.permanent = true, .deleted_buffer = true, .ready = true};
    return &event;
}

PJRT_Event *create_event(void) {
    PJRT_Event *event = calloc(1, sizeof(*event));
    if (event == NULL) {
        return NULL;
    }
    if (pthread_mutex_init(&event->lock, NULL) != 0) {
        free(event);
        return NULL;
    }
    if (pthread_cond_init(&event->completed, NULL) != 0) {
        pthread_mutex_destroy(&event->lock);
        free(event);
        return NULL;
    }
    atomic_init(&event->references, 1);
    return event;
}

void retain_event(PJRT_Event *event) {
    if (!event->permanent) {
        atomic_fetch_add_explicit(&event->references, 1, memory_order_relaxed);
    }
}

void release_event(PJRT_Event *event) {
    if (event != NULL && !event->permanent &&
        atomic_fetch_sub_explicit(&event->references, 1, memory_order_acq_rel) == 1) {
        pthread_cond_destroy(&event->completed);
        pthread_mutex_destroy(&event->lock);
        free(event);
    }
}

int await_event(PJRT_Event *event) {
    if (event->permanent) {
        return MALI_OK;
    }
    pthread_mutex_lock(&event->lock);
    while (!event->ready) {
        pthread_cond_wait(&event->completed, &event->lock);
    }
    int status = event->status;
    pthread_mutex_unlock(&event->lock);
    return status;
}

static PJRT_Error *event_result(PJRT_Event *event, int status) {
    return event->deleted_buffer
        ? make_error(PJRT_Error_Code_FAILED_PRECONDITION, "Mali buffer was deleted")
        : transfer_error(MALI_EXECUTE, status);
}

static PJRT_Error *event_destroy(PJRT_Event_Destroy_Args *args) {
    release_event(args->event);
    return NULL;
}

static PJRT_Error *event_is_ready(PJRT_Event_IsReady_Args *args) {
    PJRT_Event *event = args->event;
    if (event->permanent) {
        args->is_ready = true;
        return NULL;
    }
    pthread_mutex_lock(&event->lock);
    args->is_ready = event->ready;
    pthread_mutex_unlock(&event->lock);
    return NULL;
}

static PJRT_Error *event_error(PJRT_Event_Error_Args *args) {
    PJRT_Event *event = args->event;
    if (event->permanent) {
        return event_result(event, MALI_OK);
    }
    pthread_mutex_lock(&event->lock);
    bool ready = event->ready;
    int status = event->status;
    pthread_mutex_unlock(&event->lock);
    return ready ? event_result(event, status)
                 : make_error(PJRT_Error_Code_FAILED_PRECONDITION, "Mali event is pending");
}

static PJRT_Error *event_await(PJRT_Event_Await_Args *args) {
    return event_result(args->event, await_event(args->event));
}

static PJRT_Error *event_on_ready(PJRT_Event_OnReady_Args *args) {
    PJRT_Event *event = args->event;
    if (event->permanent) {
        args->callback(event_result(event, MALI_OK), args->user_arg);
        return NULL;
    }
    pthread_mutex_lock(&event->lock);
    if (event->ready) {
        int status = event->status;
        pthread_mutex_unlock(&event->lock);
        args->callback(event_result(event, status), args->user_arg);
        return NULL;
    }
    struct mali_callback *callback = malloc(sizeof(*callback));
    if (callback != NULL) {
        *callback = (struct mali_callback){.function = args->callback, .argument = args->user_arg};
        if (event->callback_tail == NULL) {
            event->callback_head = callback;
        } else {
            event->callback_tail->next = callback;
        }
        event->callback_tail = callback;
    }
    pthread_mutex_unlock(&event->lock);
    return callback != NULL ? NULL
        : make_error(PJRT_Error_Code_RESOURCE_EXHAUSTED, "Could not register Mali event callback");
}

void register_event_api(PJRT_Api *api) {
    api->PJRT_Event_Destroy = event_destroy;
    api->PJRT_Event_IsReady = event_is_ready;
    api->PJRT_Event_Error = event_error;
    api->PJRT_Event_Await = event_await;
    api->PJRT_Event_OnReady = event_on_ready;
}

struct mali_resource *create_resource(PJRT_Client *client, enum mali_operation release,
                                      uint64_t handle, PJRT_Event *ready) {
    struct mali_resource *resource = malloc(sizeof(*resource));
    if (resource != NULL) {
        resource->client = client;
        resource->handle = handle;
        resource->release = release;
        resource->ready = ready;
        atomic_init(&resource->references, 1);
        retain_event(ready);
    }
    return resource;
}

void retain_resource(struct mali_resource *resource) {
    atomic_fetch_add_explicit(&resource->references, 1, memory_order_relaxed);
}

void release_resource(struct mali_resource *resource) {
    if (resource == NULL || atomic_fetch_sub_explicit(&resource->references, 1,
                                                       memory_order_acq_rel) != 1) {
        return;
    }
    if (resource->handle != 0) {
        PJRT_Client *client = resource->client;
        struct mali_request request = {.operation = resource->release, .handle = resource->handle};
        struct mali_reply reply;
        pthread_mutex_lock(&client->worker_lock);
        exchange(client, &request, NULL, 0, NULL, 0, &reply);
        pthread_mutex_unlock(&client->worker_lock);
    }
    release_event(resource->ready);
    free(resource);
}

void delete_resource(PJRT_Client *client, struct mali_resource **slot) {
    pthread_mutex_lock(&client->dispatch_lock);
    struct mali_resource *resource = *slot;
    *slot = NULL;
    pthread_mutex_unlock(&client->dispatch_lock);
    release_resource(resource);
}

bool resource_deleted(PJRT_Client *client, struct mali_resource **slot) {
    pthread_mutex_lock(&client->dispatch_lock);
    bool deleted = *slot == NULL;
    pthread_mutex_unlock(&client->dispatch_lock);
    return deleted;
}

static void complete_execution(PJRT_Client *client, PJRT_Event *event, int status) {
    pthread_mutex_lock(&event->lock);
    event->status = status;
    event->ready = true;
    struct mali_callback *head = event->callback_head;
    struct mali_callback *tail = event->callback_tail;
    event->callback_head = NULL;
    event->callback_tail = NULL;
    pthread_cond_broadcast(&event->completed);
    pthread_mutex_unlock(&event->lock);

    if (head == NULL) {
        return;
    }
    for (struct mali_callback *callback = head; callback != NULL; callback = callback->next) {
        callback->status = status;
    }
    pthread_mutex_lock(&client->dispatch_lock);
    if (client->callback_tail == NULL) {
        client->callback_head = head;
    } else {
        client->callback_tail->next = head;
    }
    client->callback_tail = tail;
    pthread_cond_signal(&client->callback_available);
    pthread_mutex_unlock(&client->dispatch_lock);
}

static int execute(PJRT_Client *client, struct mali_execution *execution) {
    uint64_t handles[MALI_TRANSFER_CHUNK / sizeof(uint64_t)];
    for (size_t i = 0; i < execution->inputs; ++i) {
        struct mali_resource *input = execution->buffers[i];
        // Every pending input was published by an earlier job on this FIFO.
        int status = await_event(input->ready);
        if (status != MALI_OK) {
            return status;
        }
        handles[i] = input->handle;
    }
    struct mali_result results[MALI_MAX_OUTPUTS];
    struct mali_request request = {.operation = MALI_EXECUTE, .handle = execution->executable->handle};
    struct mali_reply reply;
    pthread_mutex_lock(&client->worker_lock);
    int status = exchange(client, &request, handles, execution->inputs * sizeof(handles[0]),
                          results, execution->outputs * sizeof(results[0]), &reply);
    pthread_mutex_unlock(&client->worker_lock);
    if (status == MALI_OK) {
        for (size_t i = 0; i < execution->outputs; ++i) {
            // Assign all handles even on a mismatch so resource cleanup can
            // release every allocation. Failed results never reach consumers.
            execution->buffers[execution->inputs + i]->handle = results[i].handle;
            if (results[i].handle == 0 || results[i].tensor.size != execution->output_sizes[i]) {
                status = MALI_RUNTIME_ERROR;
            }
        }
    }
    return status;
}

static void *dispatch_main(void *opaque) {
    PJRT_Client *client = opaque;
    for (;;) {
        pthread_mutex_lock(&client->dispatch_lock);
        while (client->work_head == NULL && !client->stop_dispatch) {
            pthread_cond_wait(&client->work_available, &client->dispatch_lock);
        }
        struct mali_execution *job = client->work_head;
        if (job == NULL) {
            pthread_mutex_unlock(&client->dispatch_lock);
            return NULL;
        }
        client->work_head = job->next;
        if (client->work_head == NULL) {
            client->work_tail = NULL;
        }
        pthread_mutex_unlock(&client->dispatch_lock);

        complete_execution(client, job->done, execute(client, job));
        for (size_t i = 0; i < job->inputs + job->outputs; ++i) {
            release_resource(job->buffers[i]);
        }
        release_resource(job->executable);
        release_event(job->done);
        free(job);
    }
}

static void destroy_dispatch(PJRT_Client *client) {
    pthread_cond_destroy(&client->callback_available);
    pthread_cond_destroy(&client->work_available);
    pthread_mutex_destroy(&client->dispatch_lock);
    free_client(client);
}

static void *callback_main(void *opaque) {
    PJRT_Client *client = opaque;
    for (;;) {
        pthread_mutex_lock(&client->dispatch_lock);
        while (client->callback_head == NULL && !client->stop_callbacks) {
            pthread_cond_wait(&client->callback_available, &client->dispatch_lock);
        }
        struct mali_callback *callback = client->callback_head;
        if (callback == NULL) {
            bool destroy = client->destroy_on_callback_exit;
            pthread_mutex_unlock(&client->dispatch_lock);
            if (destroy) {
                destroy_dispatch(client);
            }
            return NULL;
        }
        client->callback_head = callback->next;
        if (client->callback_head == NULL) {
            client->callback_tail = NULL;
        }
        pthread_mutex_unlock(&client->dispatch_lock);

        callback->function(transfer_error(MALI_EXECUTE, callback->status), callback->argument);
        free(callback);
    }
}

bool start_dispatch(PJRT_Client *client) {
    if (pthread_mutex_init(&client->dispatch_lock, NULL) != 0) {
        return false;
    }
    if (pthread_cond_init(&client->work_available, NULL) != 0) {
        goto destroy_lock;
    }
    if (pthread_cond_init(&client->callback_available, NULL) != 0) {
        goto destroy_work_signal;
    }
    if (pthread_create(&client->dispatch_thread, NULL, dispatch_main, client) != 0) {
        goto destroy_callback_signal;
    }
    if (pthread_create(&client->callback_thread, NULL, callback_main, client) == 0) {
        return true;
    }
    pthread_mutex_lock(&client->dispatch_lock);
    client->stop_dispatch = true;
    pthread_cond_signal(&client->work_available);
    pthread_mutex_unlock(&client->dispatch_lock);
    pthread_join(client->dispatch_thread, NULL);
destroy_callback_signal:
    pthread_cond_destroy(&client->callback_available);
destroy_work_signal:
    pthread_cond_destroy(&client->work_available);
destroy_lock:
    pthread_mutex_destroy(&client->dispatch_lock);
    return false;
}

// Called under dispatch_lock, after all the job's references were acquired.
void enqueue_execution(PJRT_Client *client, struct mali_execution *execution) {
    if (client->work_tail == NULL) {
        client->work_head = execution;
    } else {
        client->work_tail->next = execution;
    }
    client->work_tail = execution;
    pthread_cond_signal(&client->work_available);
}

void stop_dispatch(PJRT_Client *client) {
    pthread_mutex_lock(&client->dispatch_lock);
    client->stop_dispatch = true;
    pthread_cond_signal(&client->work_available);
    pthread_mutex_unlock(&client->dispatch_lock);
    pthread_join(client->dispatch_thread, NULL);

    // An OnReady callback can release the last client reference itself.
    bool on_callback_thread = pthread_equal(pthread_self(), client->callback_thread);
    pthread_mutex_lock(&client->dispatch_lock);
    client->stop_callbacks = true;
    client->destroy_on_callback_exit = on_callback_thread;
    pthread_cond_signal(&client->callback_available);
    pthread_mutex_unlock(&client->dispatch_lock);
    if (on_callback_thread) {
        pthread_detach(pthread_self());
        return;
    }
    pthread_join(client->callback_thread, NULL);
    destroy_dispatch(client);
}
