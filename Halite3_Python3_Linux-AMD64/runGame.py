import os
import secrets

MAX_TURNS = 50

map_settings = {32: 400,
                40: 425,
                48: 450,
                56: 475,
                64: 500}

for i in range(2):
    map_size = secrets.choice(list(map_settings.keys()))    
    commands = [f'./halite --replay-directory replays/ --no-timeout --turn-limit {MAX_TURNS} --width {map_size} -- height {map_size} "python3 MyBot-ML.py" "python3 MyBot-ML.py"',
                f'./halite --replay-directory replays/ --no-timeout --turn-limit {MAX_TURNS} --width {map_size} -- height {map_size} "python3 MyBot-ML.py" "python3 MyBot-ML.py" "python3 MyBot-ML.py" "python3 MyBot-ML.py"']

    command = secrets.choice(commands)              
    os.system(command)