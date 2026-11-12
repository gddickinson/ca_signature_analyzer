"""Start the Ca2+ Signature Analyzer web interface on http://127.0.0.1:8134 (local only).

Install: pip install -r requirements.txt -r requirements-web.txt
"""
import uvicorn

if __name__ == "__main__":
    uvicorn.run("web.app:app", host="127.0.0.1", port=8134)
