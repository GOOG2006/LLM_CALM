"""Download a remote genprm_work file. Usage: python rget.py REMOTE_BASENAME LOCAL"""
import sys, paramiko
import os
HOST = os.environ.get("REMOTE_HOST", "your.server.com")
PORT = int(os.environ.get("REMOTE_PORT", "22"))
USER = os.environ.get("REMOTE_USER", "root")
PW   = os.environ.get("REMOTE_PW", "")  # set via env; never hardcode
remote, local = sys.argv[1], sys.argv[2]
for _ in range(4):
    try:
        t = paramiko.Transport((HOST, PORT)); t.connect(username=USER, password=PW)
        s = paramiko.SFTPClient.from_transport(t)
        s.get("/root/autodl-tmp/genprm_work/" + remote, local)
        print("downloaded", local); s.close(); t.close(); break
    except Exception as e:
        import time; print("retry", e); time.sleep(6)
