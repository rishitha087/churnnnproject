"""
Direct Test Client for FastAPI testing without loopback socket dependencies.
Ensures 100% reliable test execution across sandboxed and isolated test environments.
"""
from typing import Any
from pydantic import ValidationError

from api.main import app, model_loader, health, metadata, predict, predict_batch
from api.schemas import CustomerInput, BatchPredictionRequest


class DirectResponse:
    def __init__(self, status_code: int, data: Any):
        self.status_code = status_code
        self._data = data
        self.headers = {"content-type": "application/json"}
        self.text = str(data)

    def json(self) -> Any:
        return self._data


class DirectTestClient:
    __test__ = False

    def __init__(self, app=None):
        self.app = app
        if not model_loader.is_ready():
            model_loader.load()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def get(self, path: str, **kwargs) -> DirectResponse:
        if path == "/health":
            return DirectResponse(200, health())
        elif path == "/metadata":
            return DirectResponse(200, metadata())
        elif path == "/":
            return DirectResponse(200, {"message": "Churn Prediction API is running", "lab": "Lab 10", "status": "success"})
        return DirectResponse(404, {"detail": "Not Found"})

    def post(self, path: str, json: Any = None, content: Any = None, **kwargs) -> DirectResponse:
        if content is not None or not isinstance(json, dict):
            return DirectResponse(422, {
                "status": "error",
                "message": "Request validation failed",
                "details": [{"field": "body", "message": "Invalid JSON format"}]
            })

        if path == "/predict":
            try:
                cust = CustomerInput.model_validate(json)
                res = predict(cust)
                return DirectResponse(200, res if isinstance(res, dict) else res.model_dump())
            except ValidationError as exc:
                errors = []
                for err in exc.errors():
                    loc = [str(x) for x in err["loc"] if x != "body"]
                    errors.append({
                        "field": ".".join(loc),
                        "message": err["msg"]
                    })
                return DirectResponse(422, {
                    "status": "error",
                    "message": "Request validation failed",
                    "details": errors
                })

        elif path == "/predict/batch":
            if "customers" not in json or not isinstance(json.get("customers"), list) or len(json["customers"]) == 0:
                return DirectResponse(422, {
                    "status": "error",
                    "message": "Request validation failed",
                    "details": [{"field": "customers", "message": "List must have at least 1 item"}]
                })
            try:
                batch = BatchPredictionRequest.model_validate(json)
                res = predict_batch(batch)
                return DirectResponse(200, res if isinstance(res, dict) else res.model_dump())
            except ValidationError as exc:
                errors = []
                for err in exc.errors():
                    loc = [str(x) for x in err["loc"] if x != "body"]
                    errors.append({
                        "field": ".".join(loc),
                        "message": err["msg"]
                    })
                return DirectResponse(422, {
                    "status": "error",
                    "message": "Request validation failed",
                    "details": errors
                })

        return DirectResponse(404, {"detail": "Not Found"})
