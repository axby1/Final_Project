from urllib import parse
import geoip2 as geoip2
import requests
import whois
from datetime import datetime, timezone
import math
import pandas as pd
import numpy as np
from pyquery import PyQuery
from requests import get
from rblwatch import RBLSearch
import re
from dns import resolver,reversename
import requests
import socket
import geocoder

from blacklist import virustotal



class UrlFeaturizer(object):
    def __init__(self, url):
        self.url = url
        self.domain = url.split('//')[-1].split('/')[0]
        self.today = datetime.now().replace(tzinfo=None)

        try:
            self.whois = whois.query(self.domain).__dict__
        except:
            self.whois = None

        try:
            self.response = get(self.url)
            self.pq = PyQuery(self.response.text)
        except:
            self.response = None
            self.pq = None

    ## URL string Features
    def entropy(self):
        string = self.url.strip()
        prob = [float(string.count(c)) / len(string) for c in dict.fromkeys(list(string))]
        entropy = sum([(p * math.log(p) / math.log(2.0)) for p in prob])
        return entropy

    def ip(self):
        string = self.url
        flag = False
        if ("." in string):
            elements_array = string.strip().split(".")
            if (len(elements_array) == 4):
                for i in elements_array:
                    if (i.isnumeric() and int(i) >= 0 and int(i) <= 255):
                        flag = True
                    else:
                        flag = False
                        break
        if flag:
            return 1
        else:
            return 0

    def numDigits(self):
        digits = [i for i in self.url if i.isdigit()]
        return len(digits)

    def urlLength(self):
        return len(self.url)

    def numParameters(self):
        params = self.url.split('&')
        return len(params) - 1

    def numFragments(self):
        fragments = self.url.split('#')
        return len(fragments) - 1

    def numSubDomains(self):
        subdomains = self.url.split('http')[-1].split('//')[-1].split('/')
        return len(subdomains) - 1

    def domainExtension(self):
        ext = self.url.split('.')[-1].split('/')[0]
        return ext

    ## URL domain features
    def hasHttp(self):
        return 'http:' in self.url

    def hasHttps(self):
        return 'https:' in self.url

    def daysSinceRegistration(self):
        if self.whois and self.whois['creation_date']:
            diff = self.today - self.whois['creation_date'].replace(tzinfo=None)
            diff = str(diff).split(' days')[0]
            return diff
        else:
            return 0

    def daysSinceExpiration(self):
        if self.whois and self.whois['expiration_date']:
            diff = self.whois['expiration_date'].replace(tzinfo=None) - self.today
            diff = str(diff).split(' days')[0]
            return diff
        else:
            return 0

    ## URL Page Features
    def bodyLength(self):
        if self.pq is not None:
            return len(self.pq('html').text()) if self.urlIsLive else 0
        else:
            return 0

    def numTitles(self):
        if self.pq is not None:
            titles = ['h{}'.format(i) for i in range(7)]
            titles = [self.pq(i).items() for i in titles]
            return len([item for s in titles for item in s])
        else:
            return 0

    def numImages(self):
        if self.pq is not None:
            return len([i for i in self.pq('img').items()])
        else:
            return 0

    def numLinks(self):
        if self.pq is not None:
            return len([i for i in self.pq('a').items()])
        else:
            return 0

    def scriptLength(self):
        if self.pq is not None:
            return len(self.pq('script').text())
        else:
            return 0

    def specialCharacters(self):
        if self.pq is not None:
            bodyText = self.pq('html').text()
            schars = [i for i in bodyText if not i.isdigit() and not i.isalpha()]
            return len(schars)
        else:
            return 0

    def scriptToSpecialCharsRatio(self):
        v = self.specialCharacters()
        if self.pq is not None and v != 0:
            sscr = self.scriptLength() / v
        else:
            sscr = 0
        return sscr

    def scriptTobodyRatio(self):
        v = self.bodyLength()
        if self.pq is not None and v != 0:
            sbr = self.scriptLength() / v
        else:
            sbr = 0
        return sbr

    def bodyToSpecialCharRatio(self):
        v = self.bodyLength()
        if self.pq is not None and v != 0:
            bscr = self.specialCharacters() / v
        else:
            bscr = 0
        return bscr

    def urlIsLive(self):
        return self.response == 200

    def check_tld(self):
        """Check for presence of Top-Level Domains (TLD)."""
        file = open(r'C:\Users\abbya\OneDrive\Desktop\simply\tlds.txt', 'r')
        pattern = re.compile("[a-zA-Z0-9.]")
        for line in file:
            i = (self.url.lower().strip()).find(line.strip())  # Use self.url to access the URL string
            while i > -1:
                if ((i + len(line) - 1) >= len(self.url)) or not pattern.match(
                        self.url[i + len(line) - 1]):  # Use self.url
                    file.close()
                    return 1
                i = self.url.find(line.strip(), i + 1)  # Use self.url
        file.close()
        return 0

    def count_tld(self):
        """Return amount of Top-Level Domains (TLD) present in the URL."""
        file = open(r'C:\Users\abbya\OneDrive\Desktop\simply\tlds.txt', 'r')
        count = 0
        pattern = re.compile("[a-zA-Z0-9.]")
        for line in file:
            i = (self.url.lower().strip()).find(line.strip())
            while i > -1:
                if ((i + len(line) - 1) >= len(self.url)) or not pattern.match(self.url[i + len(line) - 1]):
                    count += 1
                i = self.url.find(line.strip(), i + 1)
        file.close()
        return count

    def check_word_server_client(self):
        """Return whether the "server" or "client" keywords exist in the domain."""
        if "server" in self.url.lower() or "client" in self.url.lower():
            return 1
        return 0

    def count_name_servers(self):
        """Return number of NameServers (NS) resolved."""
        try:
            count = 0
            answers = resolver.query(self.url, 'NS')
            return len(answers) if answers else 0
        except:
            return 0
    def count_mail_servers(self):
        """Return number of NameServers (NS) resolved."""
        try:
            count = 0
            answers = resolver.query(self.url, 'MX')
            return len(answers) if answers else 0
        except:
            return 0

    def check_ssl(self):
        """Check if the SSL certificate is valid."""
        try:
            response = requests.get(self.url, verify=True, timeout=3)
            return True
        except Exception:
            return False

    def count_redirects(self):
        """Return the number of redirects in a URL."""
        try:
            response = requests.get(self.url, timeout=3)
            if response.history:
                return len(response.history) if response.history else 0
            else:
                return 0
        except Exception:
            return 0

    def get_asn_number(self):
        """Return the ANS number associated with the IP."""
        try:
            with geoip2.database.Reader(r'C:\Users\abbya\OneDrive\Desktop\simply\GeoLite2-ASN.mmdb') as reader:
                if self.check_ip():
                    ip = self.url
                else:
                    ip = resolver.query(self.url, 'A')
                    ip = ip[0].to_text()

                if ip:
                    response = reader.asn(ip)
                    return response.autonomous_system_number
                else:
                    return 0
        except Exception:
            return 0

    def ipGeolocation(self):
        try:
            # Fetch IP address of the domain
            ip_address = socket.gethostbyname(self.domain)
            # Use geocoder to get the location based on IP
            location = geocoder.ip(ip_address)
            if location:
                return location.latlng  # Return latitude and longitude
            else:
                return 0
        except Exception as e:

            return 0  ##Return 0 if no ip location found

    def get_ptr(self):
        """Return PTR associated with IP."""
        try:
            if self.check_ip():
                ip = self.url
            else:
                ip = resolver.query(self.url, 'A')
                ip = ip[0].to_text()
            if ip:
                r = reversename.from_address(ip)
                result = resolver.query(r, 'PTR')[0].to_text()
                return result
            else:
                return 0
        except Exception:
            return 0

    def check_rbl(self):
        """Check domain presence on RBL (Real-time Blackhole List)."""
        searcher = RBLSearch(self.domain)
        try:
            listed = searcher.listed
        except Exception:
            return False
        for key in listed:
            if key == 'SEARCH_HOST':
                pass
            elif listed[key]['LISTED']:
                return True
        return False

    def check_blacklists(self):
        """Check if the URL or Domain is malicious through Google Safebrowsing, Virustotal"""

        if (virustotal(self.url)!=0):
            return True
        else:
            return False

    def check_blacklists_ip(self):
        """Check if the IP is malicious through Google Safebrowsing, Virustotal"""
        try:
            if self.check_ip():
                ip = self.url
            else:
                ip = resolver.query(self.url, 'A')
                ip = ip[0].to_text()

            if ip:
                if (virustotal(ip)):
                    return True
                return False
            else:
                return False
        except Exception:
            return False

    def run(self):
        data = {}
        data['entropy'] = self.entropy()
        data['numDigits'] = self.numDigits()
        data['urlLength'] = self.urlLength()
        data['numParams'] = self.numParameters()
        data['hasHttp'] = self.hasHttp()
        data['hasHttps'] = self.hasHttps()
        data['urlIsLive'] = self.urlIsLive()
        data['bodyLength'] = self.bodyLength()
        data['numTitles'] = self.numTitles()
        data['numImages'] = self.numImages()
        data['numLinks'] = self.numLinks()
        data['scriptLength'] = self.scriptLength()
        data['specialChars'] = self.specialCharacters()
        data['ext'] = self.domainExtension()
        data['dsr'] = self.daysSinceRegistration()
        data['dse'] = self.daysSinceExpiration()
        data['sscr'] = self.scriptToSpecialCharsRatio()
        data['sbr'] = self.scriptTobodyRatio()
        data['bscr'] = self.bodyToSpecialCharRatio()
        data['num_%20'] = self.url.count("%20")
        data['num_@'] = self.url.count("@")
        data['has_ip'] = self.ip()
        data['has_tld'] = self.check_tld()
        data['tld_count'] = self.count_tld()
        data['cwsc'] = self.check_word_server_client()
        data['NS_count'] = self.count_name_servers()
        data['MX_count'] = self.count_mail_servers()
        data['has_ssl']  = self.check_ssl()
        data['redirect_count'] = self.count_redirects()
        data['asn'] = self.get_asn_number()
        data['ipgeo'] = self.ipGeolocation()
        data['ptr'] = self.get_ptr()
        data['has_rbl'] = self.check_rbl()
        data['blacklisted'] = self.check_blacklists()
        #data['blacklisted_ip'] = self.check_blacklists_ip()


        return data
