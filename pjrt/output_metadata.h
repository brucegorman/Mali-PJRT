#ifndef MALI_PJRT_OUTPUT_METADATA_H_
#define MALI_PJRT_OUTPUT_METADATA_H_

#include "iree/base/api.h"
#include "protocol.h"

bool parse_output_metadata(iree_string_view_t declaration, struct mali_executable_info *info);

#endif
