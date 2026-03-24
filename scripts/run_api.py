import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

from AdpRag.api import app
import uvicorn

if __name__ == "__main__":
    uvicorn.run("AdpRag.api:app", host="0.0.0.0", port=8000, reload=True)