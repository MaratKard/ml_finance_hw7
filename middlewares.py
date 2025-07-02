import time

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware

from metrics import app_metrics

class GetMetrics(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start = time.time()
        response = await call_next(request)
        req_time = time.time() - start
        app_name = request.app.title
        app_metrics['request_time'].labels(
            app_name=app_name,
            req_url=request.url.path,
            method=request.method,
            http_status=response.status_code
            ).observe(req_time)
        return response