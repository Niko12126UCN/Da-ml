"""Start the diabetes risk check on this PC: the API and the web UI together (step 7, local deployment).

Usage (from the project root, in the da-ml environment), or double-click start.bat:
    python run.py              start both, open the browser, Ctrl+C stops both
    python run.py --no-browser start both without opening the browser
    python run.py --check      start both, check that they respond, then stop (exit code 0 = OK)
"""

import argparse
import socket
import subprocess
import sys
import time
import urllib.request
import webbrowser
from pathlib import Path

from api.main import HOST as API_HOST, PORT as API_PORT
from ui.app import HOST as UI_HOST, PORT as UI_PORT

ROOT = Path(__file__).resolve().parent
API_URL = f"http://{API_HOST}:{API_PORT}/health"
UI_URL = f"http://{UI_HOST}:{UI_PORT}/"
START_TIMEOUT = 60   # seconds to wait for each server (D-59)


def port_in_use(host, port):
    with socket.socket() as s:
        s.settimeout(0.5)
        return s.connect_ex((host, port)) == 0


def wait_until_up(url, process, timeout=START_TIMEOUT):
    """Wait until url answers with HTTP 200. Returns False if it doesn't, or if the process exits first."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if process.poll() is not None:
            return False
        try:
            with urllib.request.urlopen(url, timeout=2) as response:
                if response.status == 200:
                    return True
        except OSError:
            pass
        time.sleep(0.5)
    return False


def stop(processes):
    for process in processes:
        if process.poll() is None:
            process.terminate()
    for process in processes:
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()


def main(argv=None):
    parser = argparse.ArgumentParser(description="Start the diabetes risk check (API + web UI).")
    parser.add_argument("--no-browser", action="store_true", help="don't open the browser")
    parser.add_argument("--check", action="store_true", help="start, check both respond, then stop")
    args = parser.parse_args(argv)

    busy = [f"{host}:{port}" for host, port in ((API_HOST, API_PORT), (UI_HOST, UI_PORT)) if port_in_use(host, port)]
    if busy:
        print(f"Port already in use: {', '.join(busy)}. Is the app already running (another window)? "
              "Close it first.")
        return 1

    processes = []
    try:
        print("Starting the API ...")
        api = subprocess.Popen([sys.executable, "-m", "api.main"], cwd=ROOT)
        processes.append(api)
        if not wait_until_up(API_URL, api):
            print("The API did not start. See the messages above.")
            return 1

        print("Starting the web UI ...")
        ui = subprocess.Popen([sys.executable, "-m", "ui.app"], cwd=ROOT)
        processes.append(ui)
        if not wait_until_up(UI_URL, ui):
            print("The web UI did not start. See the messages above.")
            return 1

        if args.check:
            print(f"OK: API ({API_URL}) and web UI ({UI_URL}) are both running.")
            return 0

        print(f"\nDiabetes risk check is running: {UI_URL}\nPress Ctrl+C in this window to stop.\n")
        if not args.no_browser:
            webbrowser.open(UI_URL)
        while all(p.poll() is None for p in processes):
            time.sleep(1)
        print("One of the servers stopped unexpectedly. See the messages above.")
        return 1
    except KeyboardInterrupt:
        print("\nStopping ...")
        return 0
    finally:
        stop(processes)


if __name__ == "__main__":
    sys.exit(main())
