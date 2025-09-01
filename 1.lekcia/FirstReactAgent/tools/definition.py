tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_temperature",
            "description": "Use this function to get the current temperature in requested city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "city": "string",
                        "description": "combined string of country and city in format country/city lowercase,  use - in country",
                    }
                },
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "convert_to_celsius",
            "description": "Use this function to convert temperature from Fahrenheit to Celsius.",
            "parameters": {
                "type": "object",
                "properties": {
                    "temp": {
                        "type": "string",
                        "description": "Temperature in Fahrenheit (e.g., '72°F' or '72')"
                    }
                },
                "required": ["temp"]
            }
        }
    },
    {
    "type": "function",
    "function": {
        "name": "get_city_forecast",
        "description": "Fetch the 7-day weather forecast for a given city from TimeAndDate.City in format country/city lowercase,  use - in country",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "The city name to get the weather forecast for, e.g., 'Prague', 'New York'."
                }
            },
            "required": ["city"]
        }
    }
    }
]