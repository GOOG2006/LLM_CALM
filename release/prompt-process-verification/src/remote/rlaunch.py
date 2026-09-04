"""Launch a remote py detached with nohup. Usage: python rlaunch.py REMOTE_PY LOGNAME"""
import sys, paramiko
import os
HOST = os.environ.get("REMOTE_HOST", "your.server.com")
PORT = int(os.environ.get("REMOTE_PORT", "22"))
USER = os.environ.get("REMOTE_USER", "root")
PW   = os.environ.get("REMOTE_PW", "")  # set via env; never hardcode
py, log = sys.argv[1], sys.argv[2]
cmd = (f"cd /root/autodl-tmp/genprm_work && rm -f {log} && "
       f"nohup /root/miniconda3/bin/python -u {py} > {log} 2>&1 & echo LAUNCHED pid=$!")
c = paramiko.SSHClient(); c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect(HOST, PORT, USER, PW, timeout=30)
i,o,e=c.exec_command(cmd)
print((o.read()+e.read()).decode()); c.close()
