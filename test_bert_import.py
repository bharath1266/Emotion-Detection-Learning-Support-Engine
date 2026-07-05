#!/usr/bin/env python
import sys
import os

print("[TEST] Python started", flush=True)
print(f"[TEST] Python version: {sys.version}", flush=True)
print(f"[TEST] CWD: {os.getcwd()}", flush=True)

sys.path.insert(0, ".")
print("[TEST] About to import training module", flush=True)

try:
    from training import train_bert
    print("[TEST] Successfully imported train_bert", flush=True)
    print("[TEST] About to call main()", flush=True)
    train_bert.main()
    print("[TEST] main() completed", flush=True)
except Exception as e:
    print(f"[TEST] ERROR: {type(e).__name__}: {e}", flush=True)
    import traceback
    traceback.print_exc()
