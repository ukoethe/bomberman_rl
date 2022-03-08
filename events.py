MOVED_LEFT = "MOVED_LEFT"
MOVED_RIGHT = "MOVED_RIGHT"
MOVED_UP = "MOVED_UP"
MOVED_DOWN = "MOVED_DOWN"
WAITED = "WAITED"
INVALID_ACTION = "INVALID_ACTION"

BOMB_DROPPED = "BOMB_DROPPED"
BOMB_EXPLODED = "BOMB_EXPLODED"

CRATE_DESTROYED = "CRATE_DESTROYED"
COIN_FOUND = "COIN_FOUND"
COIN_COLLECTED = "COIN_COLLECTED"

KILLED_OPPONENT = "KILLED_OPPONENT"
KILLED_SELF = "KILLED_SELF"

GOT_KILLED = "GOT_KILLED"
OPPONENT_ELIMINATED = "OPPONENT_ELIMINATED"
SURVIVED_ROUND = "SURVIVED_ROUND"

# Custom (Ideas)
# TODO: actually implement these
# TODO: set rewards/penalties

# Coins
DECREASED_COIN_DISTANCE = "DECREASED_COIN_DISTANCE"  # move towards nearest coing
INCREASED_COIN_DISTANCE = "INCREASED_COIN_DISTANCE"  # opposite for balance
# calculation of "coin distance" should take into consideration walls & crates (crates add some distance but don't need to be steered around?)
# penalty for moving towards bomb should be higher than reward for moving towards coin

# Navigation
STAGNATED = "STAGNATED"  # agent is still within 4-tile-radius of location 5 turns ago (4/5 bc of bomb explosion time, idk if it makes sense)
PROGRESSED = "PROGRESSED"  # opposite for balance

# Bombs
FLED = "FLED"  # was in danger zone but didn't get killed when bomb exploded
RETREATED = "REATREATED"  # increased distance towards a bomb in danger zone
SUICIDAL = "SUICIDAL"  # waited or moved towards bomb in danger zone

# Enemies
DECREASED_ENEMY_DISTANCE = "DECREASED_ENEMY_DISTANCE"  # but how do you even reward this? is it good or bad? in what situations which?
INCREASED_COIN_DISTANCE = "INCREASED_COIN_DISTANCE"  # opposite for balance
