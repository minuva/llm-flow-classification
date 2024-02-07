import uvicorn
import logging

from fastapi import FastAPI

from .controller import router
from .config import *

logger = logging.getLogger(__name__)
logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)

app = FastAPI()

app.include_router(router)

if __name__ == "__main__":
    uvicorn.run("rest_api.main:app", host=HOSTNAME, port=PORT, workers=WORKERS)
