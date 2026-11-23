#include "output_metadata.h"

#include "pjrt_c_api.h"

static bool consume(iree_string_view_t *text, const char *token) {
    *text = iree_string_view_trim(*text);
    return iree_string_view_consume_prefix(text, iree_make_cstring_view(token));
}

static bool is_digit(char character) {
    return character >= '0' && character <= '9';
}

static bool parse_type(iree_string_view_t *text, uint32_t *type) {
    static const struct {
        const char *name;
        PJRT_Buffer_Type type;
    } types[] = {
        {"i1>", PJRT_Buffer_Type_PRED},
        {"i8>", PJRT_Buffer_Type_S8}, {"si8>", PJRT_Buffer_Type_S8},
        {"i16>", PJRT_Buffer_Type_S16}, {"si16>", PJRT_Buffer_Type_S16},
        {"i32>", PJRT_Buffer_Type_S32}, {"si32>", PJRT_Buffer_Type_S32},
        {"i64>", PJRT_Buffer_Type_S64}, {"si64>", PJRT_Buffer_Type_S64},
        {"ui8>", PJRT_Buffer_Type_U8}, {"ui16>", PJRT_Buffer_Type_U16},
        {"ui32>", PJRT_Buffer_Type_U32}, {"ui64>", PJRT_Buffer_Type_U64},
        {"f16>", PJRT_Buffer_Type_F16}, {"bf16>", PJRT_Buffer_Type_BF16},
        {"f32>", PJRT_Buffer_Type_F32}, {"f64>", PJRT_Buffer_Type_F64},
        {"complex<f32>>", PJRT_Buffer_Type_C64}, {"complex<f64>>", PJRT_Buffer_Type_C128},
    };
    for (size_t i = 0; i < sizeof(types) / sizeof(types[0]); ++i) {
        if (consume(text, types[i].name)) {
            *type = types[i].type;
            return true;
        }
    }
    return false;
}

static bool parse_tensor(iree_string_view_t *text, struct mali_tensor *tensor) {
    if (!consume(text, "tensor<")) {
        return false;
    }
    while (text->size != 0 && is_digit(text->data[0])) {
        int64_t dimension = 0;
        size_t digits = 0;
        while (digits < text->size && is_digit(text->data[digits])) {
            int64_t digit = text->data[digits++] - '0';
            if (dimension > (INT64_MAX - digit) / 10) {
                return false;
            }
            dimension = dimension * 10 + digit;
        }
        if (digits == text->size || text->data[digits] != 'x' || tensor->rank == MALI_MAX_RANK) {
            return false;
        }
        tensor->dims[tensor->rank++] = dimension;
        text->data += digits + 1;
        text->size -= digits + 1;
    }
    return parse_type(text, &tensor->type);
}

bool parse_output_metadata(iree_string_view_t text, struct mali_executable_info *info) {
    // The compiler emits "sync func @main(...) -> (%output0: tensor<...>, ...)".
    const iree_string_view_t arrow = IREE_SV(") -> (");
    while (text.size >= arrow.size && memcmp(text.data, arrow.data, arrow.size) != 0) {
        ++text.data;
        --text.size;
    }
    if (!iree_string_view_consume_prefix(&text, arrow)) {
        return false;
    }
    for (uint32_t i = 0; i < info->outputs; ++i) {
        if (!consume(&text, "%")) {
            return false;
        }
        iree_host_size_t colon = iree_string_view_find_char(text, ':', 0);
        if (colon == IREE_STRING_VIEW_NPOS) {
            return false;
        }
        text.data += colon + 1;
        text.size -= colon + 1;
        struct mali_tensor *tensor = &info->output_tensors[i];
        if (!parse_tensor(&text, tensor)) {
            return false;
        }
        // IREE may canonicalize ui32 to i32 and retain its logical type here.
        if (consume(&text, "{")) {
            struct mali_tensor encoding = {0};
            if (!consume(&text, "iree.abi.encoding") || !consume(&text, "=") ||
                !parse_tensor(&text, &encoding) || !consume(&text, "}") ||
                encoding.rank != tensor->rank ||
                memcmp(encoding.dims, tensor->dims, tensor->rank * sizeof(tensor->dims[0])) != 0) {
                return false;
            }
            tensor->type = encoding.type;
        }
        if (i + 1 < info->outputs && !consume(&text, ",")) {
            return false;
        }
    }
    return consume(&text, ")") && iree_string_view_trim(text).size == 0;
}
