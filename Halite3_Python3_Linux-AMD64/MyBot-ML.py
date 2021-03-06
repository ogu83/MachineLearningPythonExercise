# Python 3.6
import hlt  #main halite stuff
from hlt import constants  # halite constants
from hlt.positionals import Direction  # helper for moving
from hlt.positionals import Position 

import random  # randomly picking a choice for now.
import logging  # logging stuff to console

import numpy as np
import time
import secrets

import os
import sys
sterr = sys.stderr
sys.stderr = open(os.devnull, "w")
import tensorflow as tf
sys.stderr = sterr    

game = hlt.Game()  # game object
# Initializes the game
game.ready("Sentdebot-ML")


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.05)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

model = tf.keras.models.load_model("models/phase1-1548223818")
RANDOM_CHANCE = secrets.choice([0.15, 0.25, 0.35])

map_settings = {32: 400,
                40: 425,
                48: 450,
                56: 475,
                64: 500}

TOTAL_TURNS = 50
# SAVE_THRESHOLD = 4100
SAVE_THRESHOLD = 0.00000001
MAX_SHIPS = 99
SIGHT_DISTANCE = 16

#specify direction order
direction_order = [Direction.North, Direction.South, Direction.East, Direction.West, Direction.Still]
    
training_data = []

initial_halite_amount = None

while True:
    # This loop handles each turn of the game. The game object changes every turn, and you refresh that state by
    game.update_frame()
    # You extract player metadata and the updated map metadata here for convenience.
    me = game.me

    '''comes from game, game comes from before the loop, hlt.Game points to networking, which is where you will 
    find the actual Game class (hlt/networking.py). From here, GameMap is imported from hlt/game_map.py. 
    
    open that file to seee all the things we do with game map.'''
    game_map = game.game_map  # game map data. Recall game is

    halite_map = np.array(list(map(lambda row: list(map(lambda cell: cell.halite_amount, row)), game_map._cells)))    
    # logging.info(halite_map)
    if initial_halite_amount is None:
        initial_halite_amount = np.sum(halite_map)

    # A command queue holds all the commands you will run this turn. You build this list up and submit it at the
    #   end of the turn.
    command_queue = []

    dropoff_positions = [d.position for d in list(me.get_dropoffs()) + [me.shipyard]]
    ship_positions = [s.position for s in list(me.get_ships())]

    for ship in me.get_ships():

        # logging.info(f"{ship.position}, {ship.position + Position(-3, 3)}")
        # logging.info(f"{game_map[ship.position + Position(-3,3)]}")

        size = SIGHT_DISTANCE 
        surroundings = []
        #surroundings = [[HALITE_AMOUNT, SHIP, DROPOFF]]

        for y in range(-1 * size, size+1):
            row=[]
            for x in range(-1 * size, size+1):
                current_cell = game_map[ship.position + Position(x, y)]

                if current_cell.position in dropoff_positions:
                    drop_friend_foe = 1
                else:
                    drop_friend_foe = -1

                if current_cell.position in ship_positions:
                    ship_friend_foe = 1
                else:
                    ship_friend_foe = -1

                halite = round(current_cell.halite_amount / constants.MAX_HALITE,2)
                a_ship = current_cell.ship
                structure = current_cell.structure

                if halite is None:
                    halite = 0

                if a_ship is None:
                    a_ship = 0
                else:
                    a_ship = round(ship_friend_foe * (a_ship.halite_amount / constants.MAX_HALITE), 2)

                if structure is None:
                    structure = 0
                else:
                    structure = drop_friend_foe

                amounts = (halite, a_ship, structure)
                row.append(amounts)                
            surroundings.append(row)

        # if game.turn_number == 5:
        #     with open("MyBot_Test.txt","w") as f:
        #         f.write(str(surroundings))

        #np.save(f"gameplay/{game.turn_number}.npy", surroundings)        
        #command_queue.append(ship.move(Direction.North))
        #command_queue.append(ship.move(random.choice([ Direction.North, Direction.South, Direction.East, Direction.West ])))
        
        if secrets.choice(range(int(1/RANDOM_CHANCE))) == 1:
            choice = secrets.choice(range(len(direction_order)))
        else:
            model_out = model.predict([np.array(surroundings).reshape(-1,len(surroundings),len(surroundings),3)])[0]
            prediction = np.argmax(model_out)
            logging.info(f"prediction: {direction_order[prediction]}")
            choice = prediction

        training_data.append([surroundings, choice])
        command_queue.append(ship.move(direction_order[choice]))

    # ship costs 1000, dont make a ship on a ship or they both sink
    if len(me.get_ships()) < MAX_SHIPS:
        if me.halite_amount >= constants.SHIP_COST and not game_map[me.shipyard].is_occupied:
            command_queue.append(me.shipyard.spawn())

    if game.turn_number == TOTAL_TURNS:
        ship_count = len(me.get_ships())
        ptc_gathered = (me.halite_amount + (ship_count * constants.SHIP_COST)) / initial_halite_amount

        if ptc_gathered >= SAVE_THRESHOLD:
            np.save(f"/media/oguz/ParrotExt/HaliteTrainingData/{ptc_gathered}-{ship_count}-{int(time.time()*1000)}.npy", training_data)

    # Send your moves back to the game environment, ending this turn.
    game.end_turn(command_queue)