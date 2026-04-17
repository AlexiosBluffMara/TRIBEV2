from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from mindscope_api.routers import jobs

app = FastAPI(
    title="MindScope API",
    description="TRIBE v2 × Gemma 4 E4B brain-response prediction API",
    version="0.1.0",
)

# CORS configuration for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health_check():
    return {"status": "ok"}


@app.get("/")
async def root():
    return {
        "message": "MindScope API",
        "version": "0.1.0",
        "docs": "/docs",
    }


# Include routers
app.include_router(jobs.router)
