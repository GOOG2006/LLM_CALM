"""Upload a local file to remote genprm_work. Usage: python rput.py LOCAL REMOTE_BASENAME"""
import sys, paramiko
import os
HOST = os.environ.get("REMOTE_HOST", "your.server.com")
PORT = int(os.environ.get("REMOTE_PORT", "22"))
USER = os.environ.get("REMOTE_USER", "root")
PW   = os.environ.get("REMOTE_PW", "")  # set via env; never hardcode
import time
local, remote = sys.argv[1], sys.argv[2]
for _ in range(6):
    try:
        t = paramiko.Transport((HOST, PORT)); t.connect(username=USER, password=PW)
        s = paramiko.SFTPClient.from_transport(t)
        s.put(local, "/root/autodl-tmp/genprm_work/" + remote)
        print("uploaded", remote); s.close(); t.close(); break
    except Exception as e:
        print("retry", e); time.sleep(6)
