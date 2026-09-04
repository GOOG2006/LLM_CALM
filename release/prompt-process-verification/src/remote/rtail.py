"""Tail last N lines of a remote file. Usage: python rtail.py REMOTE_FILE [N]"""
import sys, paramiko
import os
HOST = os.environ.get("REMOTE_HOST", "your.server.com")
PORT = int(os.environ.get("REMOTE_PORT", "22"))
USER = os.environ.get("REMOTE_USER", "root")
PW   = os.environ.get("REMOTE_PW", "")  # set via env; never hardcode
f = sys.argv[1]; n = sys.argv[2] if len(sys.argv)>2 else "15"
c = paramiko.SSHClient(); c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect(HOST, PORT, USER, PW, timeout=30)
i,o,e=c.exec_command(f"cd /root/autodl-tmp/genprm_work && tail -n {n} {f} 2>&1; echo; echo EXIT=$?")
print((o.read()+e.read()).decode()); c.close()
