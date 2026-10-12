from uuid import uuid4

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from database import init_db
from routes.runs import router as runs_router
from services.run_service import ServiceError


app = FastAPI(title="Agent Harness API", version="1.0.0")

# The teaching project uses create_all during import. A production service
# should run Alembic migrations before the application starts.
init_db()


app.include_router(runs_router)


def _request_id(request: Request) -> str:
    request_id = getattr(request.state, "request_id", None)
    if request_id is None:
        request_id = str(uuid4())
        request.state.request_id = request_id
    return request_id


def _error(
    request: Request,
    status_code: int,
    code: str,
    message: str,
) -> JSONResponse:
    request_id = _request_id(request)
    return JSONResponse(
        status_code=status_code,
        content={
            "error": {
                "code": code,
                "message": message,
                "request_id": request_id,
            }
        },
        headers={"X-Request-Id": request_id},
    )


@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    request.state.request_id = request.headers.get("X-Request-Id") or str(uuid4())
    response = await call_next(request)
    response.headers["X-Request-Id"] = request.state.request_id
    return response


@app.exception_handler(ServiceError)
async def service_error_handler(
    request: Request,
    exc: ServiceError,
) -> JSONResponse:
    return _error(request, exc.status_code, exc.code, exc.message)


@app.exception_handler(RequestValidationError)
async def validation_error_handler(
    request: Request,
    exc: RequestValidationError,
) -> JSONResponse:
    return _error(request, 422, "VALIDATION_ERROR", "Request validation failed")


@app.exception_handler(StarletteHTTPException)
async def http_error_handler(
    request: Request,
    exc: StarletteHTTPException,
) -> JSONResponse:
    code = "NOT_FOUND" if exc.status_code == 404 else "HTTP_ERROR"
    return _error(request, exc.status_code, code, str(exc.detail))


@app.exception_handler(Exception)
async def unexpected_error_handler(
    request: Request,
    exc: Exception,
) -> JSONResponse:
    return _error(request, 500, "INTERNAL_ERROR", "Internal server error")
