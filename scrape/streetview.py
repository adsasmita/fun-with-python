#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import googlemaps
import requests
import shutil
#from apiclient.discovery import build
import json
import pandas as pd
import numpy as np
import time
import os


PLACES_API_KEY = ''
SV_API_KEY = ''
headings = ['0','60','120','180','240','300']


CSV_PATH = 'csv/'
PNG_PATH = 'png/'

if not os.path.exists(CSV_PATH):
    os.makedirs(CSV_PATH)
if not os.path.exists(PNG_PATH):
    os.makedirs(PNG_PATH)
    
coor_dict = {'tokyo': {
            'lat_start':35.56,
            'lat_end':35.82,
            'lon_start':139.55,
            'lon_end':139.92}
            }

city_list = list(coor_dict.keys())

kw_list = ['lawson',
           '7-eleven',
           'family mart']

lat_start = coor_dict['tokyo']['lat_start']
lat_end = coor_dict['tokyo']['lat_end']
lon_start = coor_dict['tokyo']['lon_start']
lon_end = coor_dict['tokyo']['lon_end']

for PATH in [CSV_PATH, PNG_PATH]:
    for kw in kw_list:
        kw_path = os.path.join(PATH,kw)
        if not os.path.exists(kw_path):
            os.makedirs(kw_path)
            print(f'created {kw_path}')
                
print('keywords available:')
for i, kw in enumerate(kw_list):
    print(f'{i}: {kw}')


#radius = '2500'
keytype = 'convenience_store'
keyword = '7-eleven'
rankby = 'distance'
fov = 120
pitch = 5
headings = ['0','60','120','180','240','300']

i = 0
for lat in np.arange(lat_start, lat_end, 0.04):
    for lon in np.arange(lon_start, lon_end, 0.04):
 
        rows = [] #initialize array
        r = requests.get(f"https://maps.googleapis.com/maps/api/place/nearbysearch/json?" \
                        #+f"radius={radius}&" \
                        +f"type={keytype}&" \
                        +f"keyword={keyword}&" \
                        +f"location={lat},{lon}&" \
                        +f"rankby={rankby}&" \
                        +f"key={PLACES_API_KEY}")
        text = r.content.decode('utf-8')
        data = json.loads(text)
        results = data['results']
        for result in results:
            la, lo = result['geometry']['location'].values()

            for heading in headings:     
                r = requests.get(f"https://maps.googleapis.com/maps/api/streetview?size=640x640&" \
                                +f"location={la},{lo}&" \
                                +f"fov={fov}&" \
                                +f"heading={heading}&" \
                                +f"pitch={pitch}&" \
                                +f"key={SV_API_KEY}", stream=True)
                i += 1
                
                out_fname = f"png/{keyword}/{keyword}_{la}_{lo}_{heading}.png"
                with open(os.path.join(out_fname), 'wb') as out_file:
                    shutil.copyfileobj(r.raw, out_file)
                if i%100 ==0:
                        print(f"image saved: {i}")


        # if the result has more than 1 page, append more results
        while 'next_page_token' in data.keys():
            next_page_token = data['next_page_token']
            time.sleep(2) #wait 2 second for google activating token
            r = requests.get(f"https://maps.googleapis.com/maps/api/place/nearbysearch/json?" \
                            +f"pagetoken={next_page_token}&" \
                            +f"key={PLACES_API_KEY}")
            text = r.content.decode('utf-8')
            data = json.loads(text)
            results = data['results']

            for result in results:
                la, lo = result['geometry']['location'].values()

                for heading in headings:     
                    r = requests.get(f"https://maps.googleapis.com/maps/api/streetview?size=640x640&" \
                                    +f"location={la},{lo}&" \
                                    +f"fov={fov}&" \
                                    +f"heading={heading}&" \
                                    +f"pitch={pitch}&" \
                                    +f"key={SV_API_KEY}", stream=True)
                    
                    i += 1
                    out_fname = f"png/{keyword}/{keyword}_{la}_{lo}_{heading}.png"
                    with open(os.path.join(out_fname), 'wb') as out_file:
                        shutil.copyfileobj(r.raw, out_file)
                    
                    if i%100 ==0:
                        print(f"image saved: {i}")
