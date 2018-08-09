#!/usr/bin/env python3.6

from requests_html import HTMLSession
from subprocess import Popen, PIPE
import argparse

#html_url = "https://1337x.to/torrent/2276094/Fargo-Season-3-Complete-720p-HDTV-x264-FREDDY1714/"

ap = argparse.ArgumentParser()
ap.add_argument("-l", "--link", type=str, required=True, help="URL to webpage")
args = ap.parse_args()

html_url = args.link

def get_torrent(html_url):
    """Parse HTML to mine url to .torrent file"""
    sess = HTMLSession()
    r = sess.get(html_url)
    torrent_url = [u for u in r.html.links if u.endswith(".torrent")][0]
    torrent_fname = [f for f in torrent_url.split("/") if f.endswith(".torrent")][0]
    print(f"Obtained Torrent File URL:")
    print(f"{torrent_url}")

get_torrent(html_url)
