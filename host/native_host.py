"""Chromium Native Messaging host for read_aloud.py.

Chromium speaks length-prefixed JSON on stdin/stdout; read_aloud.py speaks JSON lines.
This relays between them, unchanged in both directions. Chromium starts the host when
the extension connects and closes stdin when it disconnects, which stops the helper and
frees its model.
"""
import json
import os
import struct
import subprocess
import sys
import threading

HERE = os.path.dirname(os.path.abspath(__file__))
# Chromium refuses host-to-browser messages over 1 MiB; the helper caps a request at 2 MiB.
MAX_EVENT = 1024 * 1024
MAX_REQUEST = 2 * 1024 * 1024
LOG = os.path.join(os.path.expanduser("~"), ".cache", "read-aloud-chromium", "helper.log")


def read_message(stream):
    """One message, or None at end of input or on a malformed frame."""
    header = stream.read(4)
    if len(header) < 4:
        return None
    (length,) = struct.unpack("=I", header)  # native byte order, per the protocol
    if length > MAX_REQUEST:
        return None
    data = stream.read(length)
    if len(data) < length:
        return None
    try:
        return json.loads(data)
    except ValueError:
        return None


def write_message(stream, message):
    data = json.dumps(message, separators=(",", ":")).encode()
    if len(data) > MAX_EVENT:
        return
    stream.write(struct.pack("=I", len(data)) + data)
    stream.flush()


def helper_command():
    return [sys.executable, "-u", os.path.join(HERE, "read_aloud.py")]


def relay(stdin, stdout, command, stderr, on_helper_exit):
    """Pump browser messages to the helper until stdin closes, then stop the helper."""
    # Never import Python modules from the browser's working directory.
    env = dict(os.environ, PYTHONSAFEPATH="1")
    child = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                             stderr=stderr, cwd=HERE, env=env)
    stopping = threading.Event()

    def events():
        for line in child.stdout:
            try:
                event = json.loads(line)
            except ValueError:
                continue
            if isinstance(event, dict):
                write_message(stdout, event)
        if not stopping.is_set():
            write_message(stdout, {"event": "error", "id": 0,
                                   "message": f"Read-aloud helper exited; see {LOG}"})
            on_helper_exit()

    reader = threading.Thread(target=events, daemon=True)
    reader.start()
    try:
        while (message := read_message(stdin)) is not None:
            child.stdin.write(json.dumps(message).encode() + b"\n")
            child.stdin.flush()
    except (BrokenPipeError, OSError):
        pass
    finally:
        stopping.set()
        child.kill()
        child.wait()
        reader.join(timeout=2)


def main():
    os.makedirs(os.path.dirname(LOG), exist_ok=True)
    with open(LOG, "w") as log:
        # A dead helper ends the host, so Chromium disconnects and the extension can retry.
        relay(sys.stdin.buffer, sys.stdout.buffer, helper_command(), log,
              on_helper_exit=lambda: os._exit(1))


if __name__ == "__main__":
    main()
