from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .api.metrics_endpoints import router as metrics_router

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Add your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add routers
app.include_router(metrics_router, prefix="/api")

# Add a test route
@app.get("/")
async def root():
    return {"message": "Water Quality Monitoring API"}