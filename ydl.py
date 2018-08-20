from subprocess import Popen, PIPE
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-f","--file", type=str, required=True, help="Path to txt file containing video URLS")

args = ap.parse_args()
txt_path = args.file
f = open(txt_path).read()
urls = f.splitlines()

def run_cmd(cmds, **kwargs):
    p = Popen(cmds, **kwargs)
    print(f"=== Executing: {' '.join(cmds)} ===")
    print("Output:")
    print(p.communicate()[0])
    print()

print(f"There are {len(urls)} VIDEOS to be downloaded")
for url in urls:
    print(f"Processing URL: {url}")
    ydl_cmds = ["youtube-dl","-f","18","-i","-o","'%(upload_date)s - %(title)s.%(ext)s'", url]
    run_cmd(ydl_cmds, stdout=PIPE, universal_newlines=True)

