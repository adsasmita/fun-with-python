#!/usr/bin/env python3.6

# Dependencies:
# pip install requests requests-html
# sudo apt install transmission-cli
from requests_html import HTMLSession
from subprocess import Popen, PIPE

### INPUTS
html_url = "https://1337x.to/torrent/2277449/Fargo-Season-3-Complete-720p-HDTV-x264-i_c/"
### END INPUTS

def get_torrent(html_url):
    """Parse HTML to mine url to .torrent file"""
    sess = HTMLSession()
    r = sess.get(html_url)
    torrent_url = [u for u in r.html.links if u.endswith(".torrent")][0]
    torrent_fname = [f for f in torrent_url.split("/") if f.endswith(".torrent")][0]
    print(f"Obtained Torrent File URL: {torrent_url}")
    print(f"Torrent Filename: {torrent_fname}")
    return torrent_url, torrent_fname

def run_cmd(cmds, **kwargs):
    p = Popen(cmds, **kwargs)
    print(f"=== Executing: {' '.join(cmds)} ===")
    print("Output:")
    print(p.communicate()[0])
    print()

torrent_url, torrent_fname = get_torrent(html_url)
wget_cmds = ["wget", torrent_url]
run_cmd(wget_cmds, stdout=PIPE, universal_newlines=True)

tcli_cmds = ["transmission-cli", torrent_fname, "-w", "."]
run_cmd(tcli_cmds, stdout=PIPE, universal_newlines=True)


