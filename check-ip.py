#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import subprocess
import time
import argparse
import smtplib

name = ''
from_address = ''
to_address = ''
subject = ''
username = ''
password = ''

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import subprocess
import time
import argparse
import smtplib

name = ''
from_address = ''
to_address = ''
subject = ''
username = ''
password = ''

ap = argparse.ArgumentParser()
ap.add_argument('-l', '--loop', default=60, type=int,
                help='Time taken for each epoch evaluation (seconds)')
args = ap.parse_args()

last_ip = '0.0.0.0'
time.sleep(60)

while True:
    ip = subprocess.Popen(['hostname', '-I'], stdout=subprocess.PIPE).stdout.read().decode()
    if ip != last_ip:
        print('New result found: {}'.format(ip))
        print('Sending Email')
        #send_email()
        msg = '\r\n'.join(['To: %s' % to_address, 'From: %s' % from_address, 'Subject: %s' % subject, '', ip])
        server = smtplib.SMTP('smtp.gmail.com:587')
        server.starttls() # Our security for transmission of credentials
        server.login(username,password)
        server.sendmail(from_address, to_address, msg)
        server.quit()
        print ("Email has been sent to: {}".format(to_address))  
        print ("Email subject: {}".format(subject))
        last_ip = ip
    time.sleep(args.loop)
