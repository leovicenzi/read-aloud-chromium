"""Hermetic host tests: a shell stub stands in for the helper; no model or audio."""
import io
import json
import os
import struct
import sys
import tempfile
import threading
import unittest

import native_host


def frame(message):
    data = json.dumps(message).encode()
    return struct.pack("=I", len(data)) + data


def frames(raw):
    out, at = [], 0
    while at + 4 <= len(raw):
        (n,) = struct.unpack("=I", raw[at:at + 4])
        out.append(json.loads(raw[at + 4:at + 4 + n]))
        at += 4 + n
    return out


STUB = r'''
import json, sys
for line in sys.stdin:
    c = json.loads(line)
    if c["cmd"] == "speak":
        print(json.dumps({"event": "progress", "id": c["id"], "chunk": 0,
                          "start": 0, "end": len(c["chunks"][0]["text"].encode())}), flush=True)
    if c["cmd"] == "quit":
        break
'''


class NativeHostTests(unittest.TestCase):
    def command(self):
        handle, path = tempfile.mkstemp(suffix=".py")
        os.write(handle, STUB.encode())
        os.close(handle)
        self.addCleanup(os.remove, path)
        return [sys.executable, path]

    def test_framing_round_trips_and_rejects_oversized_or_truncated_input(self):
        self.assertEqual(native_host.read_message(io.BytesIO(frame({"a": "olá"}))), {"a": "olá"})
        self.assertIsNone(native_host.read_message(io.BytesIO(b"\x01\x00")))
        self.assertIsNone(native_host.read_message(io.BytesIO(frame({"a": 1})[:-1])))
        huge = struct.pack("=I", native_host.MAX_REQUEST + 1)
        self.assertIsNone(native_host.read_message(io.BytesIO(huge)))
        out = io.BytesIO()
        native_host.write_message(out, {"event": "done", "id": 3})
        self.assertEqual(frames(out.getvalue()), [{"event": "done", "id": 3}])
        out = io.BytesIO()
        native_host.write_message(out, {"message": "x" * native_host.MAX_EVENT})
        self.assertEqual(out.getvalue(), b"", "oversized events are dropped, not sent broken")

    def test_relays_commands_and_events_and_stops_helper_when_browser_disconnects(self):
        read, write = os.pipe()
        out, exited = io.BytesIO(), threading.Event()
        stdin = os.fdopen(read, "rb")
        speak = {"cmd": "speak", "id": 7, "chunks": [{"text": "Olá.", "lang": "pt"}]}
        os.write(write, frame(speak))
        worker = threading.Thread(target=native_host.relay, args=(
            stdin, out, self.command(), None, exited.set))
        worker.start()
        for _ in range(500):
            if out.getvalue():
                break
            threading.Event().wait(0.01)
        os.close(write)  # the browser disconnected
        worker.join(timeout=10)
        self.assertFalse(worker.is_alive())
        self.assertEqual(frames(out.getvalue()),
                         [{"event": "progress", "id": 7, "chunk": 0, "start": 0, "end": 5}])
        self.assertFalse(exited.is_set(), "a deliberate stop is not a helper failure")

    def test_helper_exit_reports_an_error_and_ends_the_host(self):
        read, write = os.pipe()
        out, exited = io.BytesIO(), threading.Event()
        os.write(write, frame({"cmd": "quit", "id": 1}))
        worker = threading.Thread(target=native_host.relay, args=(
            os.fdopen(read, "rb"), out, self.command(), None, exited.set))
        worker.start()
        self.assertTrue(exited.wait(10))
        os.close(write)
        worker.join(timeout=10)
        [event] = frames(out.getvalue())
        self.assertEqual((event["event"], event["id"]), ("error", 0))
        self.assertIn("helper exited", event["message"])


if __name__ == "__main__":
    unittest.main()
