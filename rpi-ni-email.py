#!/usr/bin/python2
# -*- coding: utf-8 -*-
import netifaces as ni
import smtplib
import os

# Setup our login credentials
name = ''
from_address = ''
to_address = ''
subject = ''
username = ''
password = ''
path = 'ipemail/'
fname = 'last_ni.txt'

fpath = os.path.join(path, fname)

if not os.path.exists(fpath):
    os.mkdir(path)
    print ("created folder: {}".format(path))
    open(fpath, 'a').close()
    print ("created file: {}".format(fpath))
    
interfaces = ["Network Interface: {}, IPv4 Address: {}".format(i, ni.ifaddresses(i)[ni.AF_INET][0]['addr']) for i in ni.interfaces() if 'lan' in i]
body_text = '\n'.join(interfaces)
msg = '\r\n'.join(['To: %s' % to_address, 'From: %s' % from_address, 'Subject: %s' % subject, '', body_text])

# Open up previous IP address (last_ip.txt) and extract contents
with open(fpath, 'rt') as last_ip:
    last_ip = last_ip.read() # Read the text file

# Check to see if our IP address has really changed
if last_ip == body_text:
    print("Our IP address has not changed.")
else:
    print ("We have a new IP address.")
    with open(fpath, 'wt') as last_ip:
        last_ip.write(body_text)
    print ("We have written the new IP address to the text file.")
    # Actually send the email!
    server = smtplib.SMTP('smtp.gmail.com:587')
    server.starttls() # Our security for transmission of credentials
    server.login(username,password)
    server.sendmail(from_address, to_address, msg)
    server.quit()
    print ("Our email has been sent!")    
