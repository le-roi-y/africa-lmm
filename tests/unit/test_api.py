from src.api.app import create_app


class FakePredictor:
    def predict(self, question: str, top_k: int = 5):
        raise NotImplementedError


def test_create_app():
    app = create_app(FakePredictor())

    assert app.title == "AFRICA-LMM"
    assert app.version == "0.1.0"

    routes = {route.path for route in app.routes}

    assert "/health" in routes
    assert "/query" in routes
    assert "/docs" in routes
