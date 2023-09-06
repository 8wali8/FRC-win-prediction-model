import statbotics
import pandas as pd

sb = statbotics.Statbotics() 

# print(sb.get_match(match = "2019cur_qm1", fields = ["winner"]))



dict = sb.get_team_match(team = 56, match = "2019cur_qm1")

if sb.get_team_match(team = 56, match = "2019cur_qm1", fields = ["alliance"]) == sb.get_match(match = "2019cur_qm1", fields = ["winner"]):
    dict.update({"win":1})
else:
    dict.update({"win":0})


print(dict)