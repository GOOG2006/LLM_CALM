import time, paramiko, sys
import os
HOST = os.environ.get("REMOTE_HOST", "your.server.com")
PORT = int(os.environ.get("REMOTE_PORT", "22"))
USER = os.environ.get("REMOTE_USER", "root")
PW   = os.environ.get("REMOTE_PW", "")  # set via env; never hardcode
LOG=sys.argv[1]; MARK=sys.argv[2] if len(sys.argv)>2 else "DONE"; MAXW=int(sys.argv[3]) if len(sys.argv)>3 else 900
def sh(cmd):
    for _ in range(3):
        try:
            c=paramiko.SSHClient(); c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            c.connect(HOST,PORT,USER,PW,timeout=20)
            i,o,e=c.exec_command(cmd); r=(o.read()+e.read()).decode(); c.close(); return r
        except Exception as ex:
            time.sleep(5)
    return ""
t0=time.time()
while time.time()-t0<MAXW:
    r=sh(f"cd /root/autodl-tmp/genprm_work && tail -n 40 {LOG}")
    if MARK in r:
        # print result lines
        for ln in r.splitlines():
            if any(k in ln for k in ["F1=","救回","弄坏","目标","DONE","Error","Traceback"]):
                print(ln)
        print("POLL_DONE"); sys.exit(0)
    prog=[l for l in r.splitlines() if "Processed prompts" in l]
    print(f"[{int(time.time()-t0)}s] "+(prog[-1][:90] if prog else "..."), flush=True)
    time.sleep(30)
print("POLL_TIMEOUT")
