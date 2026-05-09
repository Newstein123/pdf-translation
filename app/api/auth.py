import secrets

from fastapi import Depends, HTTPException, Query, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.config import Settings, get_settings

_bearer = HTTPBearer(auto_error=True)


def require_token(
    credentials: HTTPAuthorizationCredentials = Depends(_bearer),
    settings: Settings = Depends(get_settings),
) -> None:
    if not secrets.compare_digest(credentials.credentials, settings.api_bearer_token):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="invalid bearer token"
        )


def require_token_flexible(
    request: Request,
    token: str | None = Query(default=None),
    settings: Settings = Depends(get_settings),
) -> None:
    """Accept the bearer token via Authorization header OR ``?token=`` query param.

    Used by SSE endpoints, since browsers cannot attach custom headers to
    EventSource requests. Safe for single-user local deployments; document the
    constraint elsewhere before exposing this beyond localhost.
    """
    auth = request.headers.get("authorization", "")
    presented: str | None = None
    if auth.lower().startswith("bearer "):
        presented = auth.split(" ", 1)[1].strip()
    elif token is not None:
        presented = token

    if presented is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="missing bearer token"
        )
    if not secrets.compare_digest(presented, settings.api_bearer_token):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="invalid bearer token"
        )
