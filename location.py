import requests

def get_location():
    try:
        response = requests.get('https://ipinfo.io', timeout=10)
        data = response.json()
        location = f"Lat: {data['loc'].split(',')[0]}, Lng: {data['loc'].split(',')[1]}"
        address = geolocator.reverse((data['loc'].split(',')[0], data['loc'].split(',')[1]), language='en')
        if address:
            return address.address
        return location
    except Exception as e:
        return f"Location Error: {str(e)}"
