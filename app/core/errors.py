"""Typed application errors, mapped to HTTP responses by a single handler."""


class AppError(Exception):
    status_code: int = 500
    code: str = "internal_error"

    def __init__(self, detail: str | None = None):
        self.detail = detail or self.__class__.__name__
        super().__init__(self.detail)


class NotFoundError(AppError):
    status_code = 404
    code = "not_found"


class ValidationError(AppError):
    status_code = 422
    code = "validation_error"


class UnsupportedFileType(AppError):
    status_code = 415
    code = "unsupported_file_type"


class PayloadTooLarge(AppError):
    status_code = 413
    code = "payload_too_large"


class LLMUnavailable(AppError):
    status_code = 503
    code = "llm_unavailable"


class DocumentNotReady(AppError):
    status_code = 409
    code = "document_not_ready"
