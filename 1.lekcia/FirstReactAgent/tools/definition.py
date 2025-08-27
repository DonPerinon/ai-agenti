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
                        "description": "combined string of country and city in format country/city lowercase, use - in country",
                    }
                },
                "required": ["city"],
            },
        },
    },
]