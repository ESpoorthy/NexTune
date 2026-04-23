"""
Run the NexTune Price Prediction Web App
Usage: python3 scripts/run_app.py
Then open: http://localhost:8000
"""
import uvicorn, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if __name__ == "__main__":
    uvicorn.run("src.api.app:app", host="0.0.0.0", port=8000, reload=True)
