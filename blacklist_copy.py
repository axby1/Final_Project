import configparser
import requests
import json
import requests

config = configparser.ConfigParser()
config.read('config.ini')





def virustotal(url):

    api = ""

    # VirusTotal API endpoint for URL analysis
    url_scan_url = 'https://www.virustotal.com/vtapi/v2/url/report'

    # Parameters for the API request
    params = {'apikey': api, 'resource': url}

    try:
        # Make the API request
        response = requests.get(url_scan_url, params=params)
        json_response = response.json()

        # Check if the request was successful
        if response.status_code == 200:
            # Extract and return the "Positives" value
            x=json_response['positives']
            return int(x)

        else:
            return -1

    except Exception as e:
        return -1



