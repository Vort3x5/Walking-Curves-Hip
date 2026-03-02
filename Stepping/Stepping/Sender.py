#!/usr/bin/env python3

import sys, os, tty, termios, time, subprocess

PORT_OUT = "/debug/sender"
TARGETS  = ["/robot/cmd"]

TICK = 0.02


def set_raw(fd):
    old = termios.tcgetattr(fd)
    tty.setraw(fd)
    return old


def restore(fd, old):
    termios.tcsetattr(fd, termios.TCSADRAIN, old)


def read_key_nonblock(fd):
    os.set_blocking(fd, False)
    try:
        return os.read(fd, 1)
    except BlockingIOError:
        return None
    finally:
        os.set_blocking(fd, True)


def main():
    procs = []
    for target in TARGETS:
        p = subprocess.Popen(
            ["yarp", "write", target],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )
        procs.append(p)

    def send(msg):
        for p in procs:
            try:
                p.stdin.write(msg + "\n")
                p.stdin.flush()
            except BrokenPipeError:
                pass

    print(f"  HOLD  w / s   →  idź do przodu / do tyłu")
    print(f"  TAP   a / d   →  skręt lewo / prawo")
    print(f"  q             →  wyjście")

    fd = sys.stdin.fileno()
    old_term = set_raw(fd)

    HOLD_KEYS = {b"w": "forward", b"s": "backward"}
    TAP_KEYS  = {b"a": "left",    b"d": "right"}

    held_key = None
    last_cmd = None

    try:
        while True:
            key = read_key_nonblock(fd)

            if key in (b"q", b"\x03"):
                send("stop")
                break

            elif key in HOLD_KEYS:
                if key != held_key:
                    held_key = key
                    cmd      = HOLD_KEYS[key]
                    last_cmd = cmd
                    send(cmd)
                    _status(cmd)

            elif key in TAP_KEYS:
                cmd = TAP_KEYS[key]
                send(cmd)
                _status(f"{cmd} (tap)")

            elif key is not None:
                if held_key is not None:
                    held_key = None
                if last_cmd != "stop":
                    last_cmd = "stop"
                    send("stop")
                    _status("stop")

            else:
                if held_key is None and last_cmd != "stop":
                    last_cmd = "stop"
                    send("stop")
                    _status("stop")

            time.sleep(TICK)

    finally:
        restore(fd, old_term)
        for p in procs:
            try:
                p.terminate()
            except Exception:
                pass
        print("\n[sender] bye.")


def _status(cmd):
    sys.stdout.write(f"\r[sender] -> {cmd:<20}   ")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
