import requests
from bs4 import BeautifulSoup
from datetime import datetime

def get_next_hour_temperature(city: str):
    """
    Vrátí teplotu pro první nadcházející hodinu z timeanddate.com.
    """
    city = city.lower().replace(" ", "-")
    url = f"https://www.timeanddate.com/weather/{city}/hourly"
    
    r = requests.get(url)
    if r.status_code != 200:
        return {"error": "Stránka nenalezena nebo chyba připojení."}
    
    soup = BeautifulSoup(r.text, "html.parser")
    
    table = soup.find("table", id="wt-hbh")
    if not table:
        return {"error": "Tabulka s hodinovým počasím nenalezena."}
    
    rows = table.find_all("tr")[1:]  # přeskočit hlavičku
    now_hour = datetime.now().hour
    
    for row in rows:
        hour_cell = row.find("th")
        temp_cell = row.find_all("td")

        if not hour_cell or not temp_cell:
            continue
        temp_cell = temp_cell[1]
        
        hour_text = hour_cell.text.strip().split("\n")[0]  # např. "07:00"
     
        try:
            hour_24 = int(hour_text.split(":")[0])
        except:
            continue
        
        if hour_24 >= now_hour:
            temp = temp_cell.text.split()
            return {"city": city, "hour": hour_text[:5], "temperature": temp[0]+"°C"}
    
    return {"error": "Teplota pro nadcházející hodinu nenalezena."}
