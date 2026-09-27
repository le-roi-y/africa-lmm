from __future__ import annotations

from .schemas import Document


class DocumentValidationError(ValueError):
    pass


class DocumentValidator:

    def validate(self, document: Document) -> None:
        if not document.metadata.document_id:
            raise DocumentValidationError("document_id cannot be empty.")

        if not document.metadata.filename:
            raise DocumentValidationError("filename cannot be empty.")

        if not document.text.strip() and not document.pages:
            raise DocumentValidationError("Document contains no usable content.")
