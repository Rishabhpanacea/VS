import uvicorn
from fastapi import FastAPI

from src.routers import router

app = FastAPI(debug=True)

app.include_router(router.router)
# app.include_router(prediction.router)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8004)