#include <stdlib.h>

#include "plugin_internal.h"

// Field numbers are from xla/pjrt/proto/compile_options.proto and
// DeviceAssignmentProto in the XLA revision pinned by build.py.
struct proto {
    const unsigned char *data;
    size_t remaining;
};

struct field {
    uint32_t number;
    unsigned wire;
    uint64_t value;
    struct proto payload;
};

static bool varint(struct proto *proto, uint64_t *out) {
    *out = 0;
    for (unsigned shift = 0; shift <= 63 && proto->remaining != 0; shift += 7) {
        unsigned byte = *proto->data++;
        --proto->remaining;
        if (shift == 63 && byte > 1) {
            return false;
        }
        *out |= (uint64_t)(byte & 127) << shift;
        if (byte < 128) {
            return true;
        }
    }
    return false;
}

static bool read_field(struct proto *proto, struct field *field) {
    uint64_t tag;
    if (!varint(proto, &tag) || tag < 8 || (tag >> 3) > 0x1fffffff) {
        return false;
    }
    *field = (struct field){.number = tag >> 3, .wire = tag & 7};
    uint64_t size;
    switch (field->wire) {
        case 0: return varint(proto, &field->value);
        case 1: size = 8; break;
        case 2:
            if (!varint(proto, &size)) {
                return false;
            }
            break;
        case 5: size = 4; break;
        default: return false;
    }
    if (size > proto->remaining) {
        return false;
    }
    field->payload = (struct proto){proto->data, (size_t)size};
    proto->data += size;
    proto->remaining -= size;
    return true;
}

static bool find_message(struct proto proto, uint32_t number, struct proto *out) {
    *out = (struct proto){0};
    while (proto.remaining != 0) {
        struct field field;
        if (!read_field(&proto, &field)) {
            return false;
        }
        if (field.number == number && field.wire == 2) {
            *out = field.payload;
        }
    }
    return true;
}

static int hex_digit(char character) {
    if (character >= '0' && character <= '9') {
        return character - '0';
    }
    return character >= 'a' && character <= 'f' ? character - 'a' + 10 : -1;
}

PJRT_Error *read_compile_defaults(PJRT_Client *client, const PJRT_NamedValue *options, size_t count) {
    const char name[] = "mali_compile_defaults";
    for (size_t i = 0; i < count; ++i) {
        const PJRT_NamedValue *option = &options[i];
        if (option->name_size != sizeof(name) - 1 ||
            memcmp(option->name, name, sizeof(name) - 1) != 0) {
            continue;
        }
        if (option->type != PJRT_NamedValue_kString || option->value_size == 0 ||
            option->value_size % 2 != 0) {
            return make_error(PJRT_Error_Code_INVALID_ARGUMENT, "Malformed Mali compile defaults");
        }
        size_t size = option->value_size / 2;
        unsigned char *bytes = malloc(size);
        if (bytes == NULL) {
            return make_error(PJRT_Error_Code_RESOURCE_EXHAUSTED, "Could not store Mali compile defaults");
        }
        for (size_t j = 0; j < size; ++j) {
            int high = hex_digit(option->string_value[j * 2]);
            int low = hex_digit(option->string_value[j * 2 + 1]);
            if (high < 0 || low < 0) {
                free(bytes);
                return make_error(PJRT_Error_Code_INVALID_ARGUMENT, "Malformed Mali compile defaults");
            }
            bytes[j] = (unsigned char)((high << 4) | low);
        }
        struct proto build, debug;
        if (!find_message((struct proto){bytes, size}, 3, &build) ||
            !find_message(build, 3, &debug)) {
            free(bytes);
            return make_error(PJRT_Error_Code_INVALID_ARGUMENT, "Malformed Mali compile defaults");
        }
        if (debug.remaining != 0) {
            memmove(bytes, debug.data, debug.remaining);
        }
        client->default_debug_options = bytes;
        client->default_debug_options_size = debug.remaining;
        return NULL;
    }
    return NULL;
}

static bool is_debug_map(uint32_t number) {
    // DebugOptions GPU latency estimator options and backend extra options.
    return number == 357 || number == 500;
}

static bool next_debug_field(struct proto *proto, struct proto *encoded) {
    *encoded = (struct proto){0};
    while (proto->remaining != 0) {
        struct proto before = *proto;
        struct field field;
        if (!read_field(proto, &field)) {
            return false;
        }
        // JAX changes detailed logging according to module size. It does not affect execution.
        if ((field.number == 252 && field.wire == 0 && field.value <= 1) ||
            is_debug_map(field.number)) {
            continue;
        }
        *encoded = (struct proto){before.data, before.remaining - proto->remaining};
        break;
    }
    return true;
}

static bool debug_map_counts(struct proto proto, struct proto needle,
                             size_t *total, size_t *matches) {
    *total = *matches = 0;
    while (proto.remaining != 0) {
        struct proto before = proto;
        struct field field;
        if (!read_field(&proto, &field)) {
            return false;
        }
        if (is_debug_map(field.number)) {
            if (field.wire != 2) {
                return false;
            }
            ++*total;
            size_t size = before.remaining - proto.remaining;
            if (size == needle.remaining && memcmp(before.data, needle.data, size) == 0) {
                ++*matches;
            }
        }
    }
    return true;
}

static bool same_debug_maps(struct proto actual, struct proto baseline) {
    struct proto remaining = actual;
    size_t actual_total = 0, baseline_total, matches;
    if (!debug_map_counts(baseline, (struct proto){0}, &baseline_total, &matches)) {
        return false;
    }
    while (remaining.remaining != 0) {
        struct proto before = remaining;
        struct field field;
        if (!read_field(&remaining, &field)) {
            return false;
        }
        if (is_debug_map(field.number)) {
            ++actual_total;
            struct proto needle = {before.data, before.remaining - remaining.remaining};
            size_t actual_count, baseline_count, total;
            if (!debug_map_counts(actual, needle, &total, &actual_count) ||
                !debug_map_counts(baseline, needle, &total, &baseline_count) ||
                actual_count != baseline_count) {
                return false;
            }
        }
    }
    return actual_total == baseline_total;
}

static bool default_debug_options(const PJRT_Client *client, struct proto actual) {
    struct proto baseline = {client->default_debug_options, client->default_debug_options_size};
    if (!same_debug_maps(actual, baseline)) {
        return false;
    }
    while (actual.remaining != 0 || baseline.remaining != 0) {
        struct proto left, right;
        if (!next_debug_field(&actual, &left) || !next_debug_field(&baseline, &right) ||
            left.remaining != right.remaining ||
            (left.remaining != 0 && memcmp(left.data, right.data, left.remaining) != 0)) {
            return false;
        }
    }
    return true;
}

static bool assigned_device(struct proto devices) {
    size_t count = 0;
    while (devices.remaining != 0) {
        struct field field;
        if (!read_field(&devices, &field) || field.number != 1) {
            return false;
        }
        if (field.wire == 0 && field.value == 0) {
            ++count;
        } else if (field.wire == 2) {
            struct proto packed = field.payload;
            while (packed.remaining != 0) {
                uint64_t ordinal;
                if (!varint(&packed, &ordinal) || ordinal != 0) {
                    return false;
                }
                ++count;
            }
        } else {
            return false;
        }
    }
    return count == 1;
}

static bool single_device_assignment(struct proto assignment) {
    bool one_replica = false, one_computation = false;
    size_t entries = 0;
    while (assignment.remaining != 0) {
        struct field field;
        if (!read_field(&assignment, &field)) {
            return false;
        }
        if (field.number == 1 && field.wire == 0) {
            one_replica = field.value == 1;
        } else if (field.number == 2 && field.wire == 0) {
            one_computation = field.value == 1;
        } else if (field.number == 3 && field.wire == 2 && assigned_device(field.payload)) {
            ++entries;
        } else {
            return false;
        }
    }
    return one_replica && one_computation && entries == 1;
}

static bool valid_bool_list(const struct field *field) {
    if (field->wire == 0) {
        return field->value <= 1;
    }
    if (field->wire != 2) {
        return false;
    }
    struct proto packed = field->payload;
    while (packed.remaining != 0) {
        uint64_t value;
        if (!varint(&packed, &value) || value > 1) {
            return false;
        }
    }
    return true;
}

static const char *check_build_options(const PJRT_Client *client, struct proto build) {
    while (build.remaining != 0) {
        struct field field;
        if (!read_field(&build, &field)) {
            return "Malformed XLA executable build options";
        }
        switch (field.number) {
            case 1:
                if (field.wire != 0 || (field.value != 0 && field.value != UINT64_MAX)) {
                    return "Mali only supports device ordinal 0";
                }
                break;
            case 4:
            case 5:
                if (field.wire != 0 || field.value != 1) {
                    return "Mali requires one replica and one partition";
                }
                break;
            case 9:
                if (field.wire != 2 || !single_device_assignment(field.payload)) {
                    return "Mali requires a single-device assignment to device 0";
                }
                break;
            case 22:
            case 23:
                if (field.wire != 0 || field.value != (uint64_t)(field.number == 23)) {
                    return "Mali supports only one local process";
                }
                break;
            case 3:
                if (field.wire != 2) {
                    return "Malformed XLA debug options";
                }
                if (!default_debug_options(client, field.payload)) {
                    return "Mali does not support custom XLA debug options";
                }
                break;
            case 6:
            case 19:
                // JAX enables these flags even for one partition. They have no effect here.
                if (field.wire != 0 || field.value > 1) {
                    return "Malformed XLA partitioning flag";
                }
                break;
            case 12:
            case 18:
                // Sharding propagation is inert on one device.
                if (!valid_bool_list(&field)) {
                    return "Malformed XLA sharding options";
                }
                break;
            case 25:
                // JAX 0.6.1 serializes its default EFFORT_O2 (29).
                if (field.wire != 0 || (field.value != 0 && field.value != 29)) {
                    return "Mali does not support a custom memory fitting level";
                }
                break;
            default:
                return "Unsupported XLA executable build option on Mali";
        }
    }
    return NULL;
}

PJRT_Error *validate_compile_options(const PJRT_Client *client, const char *bytes, size_t size) {
    if (size != 0 && bytes == NULL) {
        return make_error(PJRT_Error_Code_INVALID_ARGUMENT, "Missing XLA compile options");
    }
    struct proto options = {(const unsigned char *)bytes, size};
    while (options.remaining != 0) {
        struct field field;
        if (!read_field(&options, &field)) {
            return make_error(PJRT_Error_Code_INVALID_ARGUMENT, "Malformed XLA compile options");
        }
        if (field.number == 3 && field.wire == 2) {
            const char *error = check_build_options(client, field.payload);
            if (error != NULL) {
                return make_error(PJRT_Error_Code_INVALID_ARGUMENT, error);
            }
        } else if (field.number == 5 && field.wire == 0 &&
                   (field.value == 0 || field.value == UINT64_MAX)) {
            continue;
        } else if (field.number == 7) {
            return make_error(PJRT_Error_Code_UNIMPLEMENTED,
                              "Mali does not support XLA compiler option overrides");
        } else {
            return make_error(PJRT_Error_Code_UNIMPLEMENTED, "Unsupported XLA compile option on Mali");
        }
    }
    return NULL;
}
