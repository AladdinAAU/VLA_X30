import socket, struct, time
s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM); s.bind(("127.0.0.1", 43893)); s.settimeout(0.5)
t0 = time.time(); last = None; gaps = []
while time.time() - t0 < 9:
    try: d, a = s.recvfrom(2048)
    except socket.timeout: continue
    c = struct.unpack("<I", d[:4])[0]
    now = time.time()
    if c == 0x21040001:
        if last: gaps.append(now - last)
        last = now
    else: print(f"{now - t0:.3f} rx {c:08X}", flush=True)
import collections
h = collections.Counter(round(g, 2) for g in gaps)
print(len(gaps) + 1, "hb; gaps:", sorted(h.items())[:12])
