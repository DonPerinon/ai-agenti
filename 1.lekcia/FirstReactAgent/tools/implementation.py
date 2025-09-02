import requests
from bs4 import BeautifulSoup
from datetime import datetime

def parse_hour(hour_text):
    hour_text = hour_text.strip()
    for fmt in ("%I:%M %p", "%H:%M"):  # 12-hour and 24-hour
        try:
            dt = datetime.strptime(hour_text, fmt)
            return dt.hour
        except ValueError:
            continue
    return None

def get_next_hour_temperature(city: str):
    """
    Vrátí teplotu pro první nadcházející hodinu z timeanddate.com.
    """
    city = city.lower().replace(" ", "-")
    url = f"https://www.timeanddate.com/weather/{city}/hourly"
    
    r = requests.get(url)
    if r.status_code != 200:
        if r.status_code == 404:
            return {"error": "Possible wrong country format"}
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
        hour_text = hour_cell.text.strip().split("\n")[0]  # např. "07:00" nebo "1:00 PM"
        
        hour_24 = parse_hour(hour_text)
        if hour_24 is None:
            continue

        if hour_24 >= now_hour:
            temp = temp_cell.text.split()
            return {"city": city, "hour": hour_text[:5], "temperature": temp[0], "unit": temp[1]}
    
    return {"error": "Teplota pro nadcházející hodinu nenalezena."}

def convert_to_celsius(temp: str) -> str:
    temp_num = ''.join(c for c in temp if c.isdigit() or c == '.' or c == '-')
    
    try:
        f = float(temp_num)
    except ValueError:
        return "Invalid temperature input"

    c = (f - 32) * 5 / 9
    return f"{c:.1f}°C"

def get_city_forecast(city: str):
    url = f"https://www.timeanddate.com/weather/{city}/ext"
    
    r = requests.get(url)
    if r.status_code != 200:
        if r.status_code == 404:
            return {"error": "Possible wrong country format"}
        return {"error": "Stránka nenalezena nebo chyba připojení."}
    
    soup = BeautifulSoup(r.text, "html.parser")
    
    # Find the 14-day forecast table
    forecast_table = soup.find("table", attrs={"id": "wt-ext"})
    if not forecast_table:
        return {"error": "Tabulka s předpovědí nenalezena."}
    
    forecast = []
    rows = forecast_table.find_all("tr")[1:]  # skip header
    
    for row in rows:
        # Day info is in <th>
        day_cell = row.find("th")
        if not day_cell:
            continue

        # Weekday (e.g., Tue)
        weekday_span = day_cell.find("span", class_="smaller soft")
        weekday = weekday_span.text.strip() if weekday_span else ""

        # Date (e.g., Sep 2)
        day_text_nodes = [node for node in day_cell.contents if isinstance(node, str)]
        date_text = day_text_nodes[0].strip() if day_text_nodes else ""
        
        day_info = f"{weekday} {date_text}"  # e.g., "Tue Sep 2"

        # Other forecast details in <td>
        cells = row.find_all("td")
        if len(cells) < 5:
            continue

        description = cells[0].text.strip()
        temp_high = cells[1].text.strip()
        temp_low = cells[2].text.strip()
        wind = cells[3].text.strip()
        humidity = cells[4].text.strip()

        forecast.append({
            "day": day_info,
            "description": description,
            "high": temp_high,
            "low": temp_low,
            "wind": wind,
            "humidity": humidity
        })
    
    return forecast