#!/usr/bin/python2
# -*- coding: utf-8 -*-
import netifaces as ni
import smtplib

# Setup our login credentials
name = ''
from_address = ''
to_address = ''
subject = ''
username = ''
password = ''

interfaces = ["Network Interface: {}, IPv4 Address: {}".format(i, ni.ifaddresses(i)[ni.AF_INET][0]['addr']) for i in ni.interfaces() if 'lan' in i]
body_text = '\n'.join(interfaces)
msg = '\r\n'.join(['To: %s' % to_address, 'From: %s' % from_address, 'Subject: %s' % subject, '', body_text])

# Actually send the email!
server = smtplib.SMTP('smtp.gmail.com:587')
server.starttls() # Our security for transmission of credentials
server.login(username,password)
server.sendmail(from_address, to_address, msg)
server.quit()
print ("Our email has been sent!")
