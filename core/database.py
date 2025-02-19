import json
import os

def collect_data(command, response):
    data = {"command": command, "response": response}
    if not os.path.exists("data"):
        os.makedirs("data")
    with open("data/commands.json", "a") as f:
        json.dump(data, f)
        f.write("\n")

def load_data():
    data = []
    if os.path.exists("data/commands.json"):
        with open("data/commands.json", "r") as f:
            for line in f:
                try:
                    data.append(json.loads(line.strip()))
                except json.JSONDecodeError:
                    continue
    return data