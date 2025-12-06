import sys
from pathlib import Path

from litestar import Litestar, get
from litestar.config.cors import CORSConfig
from litestar.static_files import StaticFilesConfig
from litestar.status_codes import HTTP_200_OK
from loguru import logger

# Configure logger
logger.remove()
logger.add(sys.stderr, level="DEBUG")

from .routers import AnalysisController

# Static files configuration
static_dir = Path.cwd() / "web" / "dist"
static_files_config = []
if static_dir.exists():
    static_files_config.append(
        StaticFilesConfig(directories=[static_dir], path="/", html_mode=True)
    )

# CORS configuration
cors_config = CORSConfig(allow_origins=["*"])


@get("/health", status_code=HTTP_200_OK)
async def health_check() -> dict[str, str]:
    return {"status": "ok"}


app = Litestar(
    route_handlers=[health_check, AnalysisController],
    cors_config=cors_config,
    static_files_config=static_files_config,
    debug=True,
)
