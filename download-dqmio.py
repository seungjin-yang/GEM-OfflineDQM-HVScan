import os
import httplib
import ssl
import argparse
import urllib2

HTTPS = httplib.HTTPSConnection
CONTEXT = ssl._create_unverified_context()

# assume that your ca files exist under ~/.globus
HOME = os.getenv("HOME")
GLOBUS = os.path.join(HOME, ".globus")
SSL_KEY_FILE = os.path.join(GLOBUS, "userkey.pem")
SSL_CERT_FILE = os.path.join(GLOBUS, "usercert.pem")

class HTTPSCertAuth(HTTPS):
    def __init__(self, host, *args, **kwargs):
        HTTPS.__init__(self,
                       host,
                       key_file=SSL_KEY_FILE,
                       cert_file=SSL_CERT_FILE,
                       context=CONTEXT,
                       **kwargs)

class HTTPSCertAuthHandler(urllib2.AbstractHTTPHandler):
    def default_open(self, req):
        return self.do_open(HTTPSCertAuth, req)



PREFIX = "https://cmsweb.cern.ch/dqm/offline/data/browse/ROOT/"
DEST = "/store/scratch/dqm/"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", type=str, required=True)
    args = parser.parse_args()

    request = urllib2.Request(args.url)
    handler = HTTPSCertAuthHandler()
    result = urllib2.build_opener(handler).open(request)
    data  = result.read()

    filename = os.path.basename(args.url)

    with open(filename, "w") as root_file:
        root_file.write(data)

if __name__ == "__main__":
    main()
