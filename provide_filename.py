from __future__ import annotations

import importlib
import subprocess
import sys
from pathlib import Path
from typing import Optional

INTEGRATION = "import"   
TARGET_MODULE = "emotion_infer"     
TARGET_FUNCTION = "process_file"   
TARGET_SCRIPT = "emotion_infer"  
EXTRA_ARGS: list[str] = []    

def _coerce_path(s: str) -> Path:
    s = s.strip().strip('"').strip("'")
    return Path(s).expanduser()

def _prompt_for_path() -> Path:
    while True:
        raw = input("Enter the path to the input file: ").strip()
        if not raw:
            print("Please type a path (or sned an interrupt to cancel)." )
            continue
        path = _coerce_path(raw)
        if path.exists():
            return path
        print(f"Not found: {path}. Try again.")

def get_input_path(arg: Optional[str]) -> Path:
    if arg:
        p = _coerce_path(arg)
        if not p.exists():
            print(f"File not found: {p}")
            return _prompt_for_path()
        return p
    return _prompt_for_path()

def run_with_import(path: Path) -> None:
    try:
        mod = importlib.import_module(TARGET_MODULE)
    except Exception as e:
        print(f"Could not import module '{TARGET_MODULE}': {e}")
        sys.exit(1)
    try:
        func = getattr(mod, TARGET_FUNCTION)
    except AttributeError:
        print(f"Module '{TARGET_MODULE}' has no function '{TARGET_FUNCTION}'.")
        sys.exit(1)
    try:
        result = func(str(path))
        if result is not None:
            print("Function returned:", result)
    except Exception as e:
        print(f"Error while calling {TARGET_MODULE}.{TARGET_FUNCTION}('{path}'):\n{e}")
        sys.exit(1)

def run_with_subprocess(path: Path) -> None:
    cmd = [sys.executable, TARGET_SCRIPT, str(path), *EXTRA_ARGS]
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Script exited with non-zero status {e.returncode}.")
        sys.exit(e.returncode)
    except FileNotFoundError:
        print(f"Script '{TARGET_SCRIPT}' not found.")
        sys.exit(1)

def main(argv: list[str]) -> None:
    arg = argv[1] if len(argv) > 1 else None
    path = get_input_path(arg)
    if INTEGRATION == "import":
        run_with_import(path)
    elif INTEGRATION == "subprocess":
        run_with_subprocess(path)
    else:
        print("INTEGRATION must be either 'import' or 'subprocess'.")
        sys.exit(1)

if __name__ == "__main__":
    main(sys.argv)
