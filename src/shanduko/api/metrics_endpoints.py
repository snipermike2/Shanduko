from fastapi import APIRouter, HTTPException
from pathlib import Path
import json
from datetime import datetime

router = APIRouter()

@router.get("/metrics/latest")
async def get_latest_metrics():
    """Get the latest metrics from the metrics directory"""
    try:
        metrics_dir = Path("reports/metrics")
        if not metrics_dir.exists():
            raise HTTPException(status_code=404, detail="No metrics data found")  # Changed 'message' to 'detail'
            
        # Get latest metrics file
        metrics_files = list(metrics_dir.glob("metrics_*.json"))
        if not metrics_files:
            raise HTTPException(status_code=404, detail="No metrics data found")  # Changed 'message' to 'detail'
            
        latest_file = max(metrics_files, key=lambda x: x.stat().st_mtime)
        
        # Read metrics
        with open(latest_file) as f:
            metrics_data = json.load(f)
            
        return metrics_data
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Adding a test endpoint
@router.get("/metrics/test")
async def test_metrics():
    """Test endpoint to verify API is working"""
    return {
        "status": "ok",
        "timestamp": datetime.now().isoformat(),
        "message": "Metrics API is working"
    }