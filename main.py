import os
import runpy
import subprocess
import sys


def _running_in_streamlit() -> bool:
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx

        return get_script_run_ctx(suppress_warning=True) is not None
    except Exception:
        return False


if __name__ == "__main__" and not _running_in_streamlit():
    app_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "app.py")
    try:
        subprocess.run(
            [sys.executable, "-m", "streamlit", "run", app_path],
            check=False,
        )
    except KeyboardInterrupt:
        pass
    sys.exit(0)

if __name__ == "__main__":
    app_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "app.py")
    runpy.run_path(app_path)
