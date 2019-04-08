"""
   #Return number of legal moves by player
   return float(len(game.get_legal_moves(player)))
"""

"""
# Gets the current position of the player
position = game.get_player_location(player)
# Calculates distance to the center
return -float(math.sqrt((position[1] + 1 - game.width / 2.0) ** 2 + (position[0] + 1 - game.height / 2.0) ** 2))

"""
"""
# Gets the current position of the player
position = game.get_player_location(player)
# Calculates distance to the center
return float(min([position[1], game.width - (position[1] + 1), position[0], game.height - (position[0] + 1)]))
"""