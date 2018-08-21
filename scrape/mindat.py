#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#Website: mindat.org (mineral data)

import os
import tqdm
import requests
from requests_html import HTMLSession

### INPUTS ###
IS_FOLDERS = True
DEBUG = True
ROOT_URL = 'https://www.mindat.org/'
### END INPUTS ###

def printf(s):
    if DEBUG: print(s)

def loc_path(url): 
    """Local path without https url"""
    return url[len(ROOT_URL):] if IS_FOLDERS else url.split('/')[-1]

def mkdir_loc(url):
    """Create sub-folders in local directory."""
    if IS_FOLDERS and not os.path.exists(loc_path(url)):
        os.mkdir(loc_path(url))
        printf(f"Created folder at {loc_path(url)}")

def try_http_get(url, **kwargs):
    try:
        headers = kwargs.get('headers')
        r = sess.get(url, headers=headers)
        return r
    except requests.exceptions.RequestException as e:
        print(e, url)
        return None

# create HTML session
sess = HTMLSession()

# first parent
p1 = ROOT_URL+'imagecache/'
mkdir_loc(p1)
r = try_http_get(p1)
printf(f"parent url: {p1}, status code: {r.status_code}")

# first child
c1s = sorted([p1+u for u in list(r.html.links) if len(u) == 3])
for c1 in tqdm.tqdm(c1s):
    mkdir_loc(c1)
    r = try_http_get(c1, headers={'referer': '/'.join(c1.split('/')[:-1])})
    if r is None: continue
    printf(f"first child url: {c1}, status code: {r.status_code}")

    # second child
    c2s = sorted([c1+u for u in list(r.html.links) if len(u) == 3])
    for c2 in tqdm.tqdm(c2s):
        mkdir_loc(c2)
        r = try_http_get(c2, headers={'referer': '/'.join(c2.split('/')[:-1])})
        if r is None: continue
        printf(f"second child url: {c2}, status code: {r.status_code}")
    
        # third child
        c3s = [c2+u for u in list(r.html.links) if '.jpg' in u or '.png' in u]
        for c3 in c3s:
            # get JPG HTML response
            r = try_http_get(c3, headers={'referer': '/'.join(c3.split('/')[:-1])})
            if r is None: continue
            printf(f"third child url: {c3}, status code: {r.status_code}")
            # save image
            if r.status_code == 200 and r.content != b'':
                jpg_path = loc_path(c3)
                with open(jpg_path, 'wb') as f:
                    f.write(r.html.raw_html)
