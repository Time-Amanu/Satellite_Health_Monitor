import requests
def fetch_tle_data():
    url = 'https://celestrak.org/NORAD/elements/gp.php?GROUP=active&FORMAT=tle'
    response = requests.get(url, timeout=30)
    lines = response.text.strip().split('
')
    return [l.strip() for l in lines if l.strip()]