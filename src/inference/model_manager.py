from __future__ import annotations

from threading import Lock

from src.models.base import BaseModel


class ModelManager:
    """Gestionnaire du cycle de vie du modèle d'inférence."""

    def __init__(self, model: BaseModel) -> None:
        self.model = model
        self._lock = Lock()

    @property
    def loaded(self) -> bool:
        return self.model.loaded

    def load(self) -> None:
        """Charge le modèle une seule fois."""
        if self.model.loaded:
            return

        with self._lock:
            if not self.model.loaded:
                self.model.load()

    def unload(self) -> None:
        """Décharge le modèle."""
        with self._lock:
            if self.model.loaded:
                self.model.unload()

    def get_model(self) -> BaseModel:
        """Retourne le modèle après s'être assuré qu'il est chargé."""
        self.load()
        return self.model
