#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SAMPLE USAGE
# python ~/pyscripts/ydl.py -s list.txt -f 22
# 18 or 22 for video, 140 or 251 for audio
import argparse
from subprocess import Popen, PIPE

ap = argparse.ArgumentParser()
ap.add_argument("-s","--source", type=str, required=True, help="Path to txt file containing video URLS")
ap.add_argument("-f","--format", type=int, default=18, help="File format (18 or 22 for video, 140 or 251 for audio)")

args = ap.parse_args()
txt_path = args.source
file_format = str(args.format)

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
    ydl_cmds = ['youtube-dl','-f',file_format,'-i','-o','%(upload_date)s - %(title)s.%(ext)s', url]
    run_cmd(ydl_cmds, stdout=PIPE, universal_newlines=True)

