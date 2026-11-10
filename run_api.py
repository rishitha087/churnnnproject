import time
import webbrowser
import threading
import uvicorn

def open_dashboard():
    time.sleep(1.5)
    dashboard_url = "http://127.0.0.1:8000/docs"
    print(f"\n[INFO] Opening Churn API Dashboard in browser: {dashboard_url}\n")
    try:
        webbrowser.open(dashboard_url)
    except Exception as exc:
        print(f"[WARN] Could not automatically open browser: {exc}")

if __name__ == "__main__":
    print("=" * 65)
    print(" Churn Prediction Service & Dashboard ")
    print(" API Base URL   : http://127.0.0.1:8000")
    print(" Swagger UI URL : http://127.0.0.1:8000/docs")
    print(" Health Endpoint: http://127.0.0.1:8000/health")
    print("=" * 65)
    print("Press CTRL+C to stop the server.\n")

    threading.Thread(target=open_dashboard, daemon=True).start()
    uvicorn.run("api.main:app", host="127.0.0.1", port=8000, log_level="info")

