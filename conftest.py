"""pytest: setup.py's data-folder migration (#198) saves the data folder it settled on in the user's settings file
(%APPDATA%\Strata\settings.json, ~/.config/strata/settings.json) and can MOVE model files from a folder it finds there.
A test that reaches it must never touch the real file, so the whole session gets its own config home."""
import os
import tempfile

_home = tempfile.mkdtemp(prefix="strata-pytest-config-")
os.environ["APPDATA"] = _home
os.environ["XDG_CONFIG_HOME"] = _home

# tools/*_test.py are scripts run by hand against a live server (python3 tools/x_test.py CFG|URL), not pytest tests:
# collected, the ones that read sys.argv or a config at import fail before any test runs
collect_ignore_glob = ["tools/*_test.py"]
