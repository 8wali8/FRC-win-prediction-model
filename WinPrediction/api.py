import statbotics
import pandas as pd

sb = statbotics.Statbotics()
print(sb.get_team(254,["wins", "ties"]))