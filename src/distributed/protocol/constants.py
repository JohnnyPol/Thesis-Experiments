from __future__ import annotations

# Protocol versioning
PROTOCOL_VERSION = "1.0"


# Request kinds
REQUEST_KIND_INPUT = "input"
REQUEST_KIND_ACTIVATION = "activation"


# Final / terminal statuses
RESPONSE_STATUS_EXITED = "exited"
RESPONSE_STATUS_COMPLETED = "completed"
RESPONSE_STATUS_ERROR = "error"


# Internal tensor layout names
TENSOR_LAYOUT_NCHW = "NCHW"


# MIME / transport helpers
CONTENT_TYPE_MULTIPART = "multipart/form-data"
CONTENT_TYPE_BINARY = "application/octet-stream"
METADATA_FORM_FIELD = "metadata"
TENSOR_FORM_FIELD = "tensor_file"


# Default tensor serialization choices
DEFAULT_TENSOR_DTYPE = "float32"
DEFAULT_TIMEOUT_SEC = 30.0


# Valid value sets
VALID_REQUEST_KINDS = {
    REQUEST_KIND_INPUT,
    REQUEST_KIND_ACTIVATION,
}

VALID_RESPONSE_STATUSES = {
    RESPONSE_STATUS_EXITED,
    RESPONSE_STATUS_COMPLETED,
    RESPONSE_STATUS_ERROR,
}

VALID_TENSOR_LAYOUTS = {
    TENSOR_LAYOUT_NCHW,
}