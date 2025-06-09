import requests

def get_location():
    try:
        response = requests.get('http://ipinfo.io')
        data = response.json()
        location = data['loc'].split(',')
        city = data.get('city', 'Unknown')
        country = data.get('country', 'Unknown')
        ip_address = data.get('ip', 'Unknown')  # Get the IP address

        return {
            'ip_address': ip_address,
            'latitude': location[0],
            'longitude': location[1],
            'city': city,
            'country': country
        }
    except Exception as e:
        return {
            'ip_address': 'Unknown',
            'latitude': 'Unknown',
            'longitude': 'Unknown',
            'city': 'Unknown',
            'country': 'Unknown'
        }
