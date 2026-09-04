"""Run a remote python file already uploaded to genprm_work. Usage: python rrun.py REMOTE_PY [args...]"""
import sys, paramiko, time
import os
HOST = os.environ.get("REMOTE_HOST", "your.server.com")
PORT = int(os.environ.get("REMOTE_PORT", "22"))
USER = os.environ.get("REMOTE_USER", "root")
PW   = os.environ.get("REMOTE_PW", "")  # set via env; never hardcode
pyfile = sys.argv[1]; args = " ".join(sys.argv[2:])
cmd = f"cd /root/autodl-tmp/genprm_work && /root/miniconda3/bin/python -u {pyfile} {args}"
for _ in range(4):
    try:
        c = paramiko.SSHClient(); c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        c.connect(HOST, PORT, USER, PW, timeout=30)
        stdin, stdout, stderr = c.exec_command(cmd, get_pty=True)
        for line in iter(stdout.readline, ""):
            sys.stdout.write(line); sys.stdout.flush()
        err = stderr.read().decode(errors="ignore")
        if err.strip(): sys.stderr.write(err)
        c.close(); break
    except Exception as e:
        print("retry", e); time.sleep(6)
