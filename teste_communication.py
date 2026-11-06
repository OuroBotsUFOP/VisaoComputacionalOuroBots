import sys
import subprocess
import time


process = subprocess.Popen(
    ["./.venv/Scripts/python.exe", "./src/main.py"],
    stdout = subprocess.PIPE,
    text= True,
    bufsize=1
)

while True:
    output= process.stdout.readline()
    if not output:
        break
    print(output)
    time.sleep(1/60)