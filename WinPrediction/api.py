import statbotics
import pandas as pd

sb = statbotics.Statbotics() 

#names of teams on alliance
alliance = ["red_1", "red_2", "red_3", "blue_1", "blue_2", "blue_3"]

#finding number of events
events = sb.get_events(year = 2023, fields=["key"])
dictionary = []
numevent = 0

#iterates through events
for y in events:
    numevent = numevent + 1
    print("event number: " + str(numevent) + "\n")
    print("event name: " + str(y['key'] + "\n"))
    z=0
    #iterates through matches in events
    for x in (sb.get_matches(event = str(y['key']))):
        nummatch = x.get("key")
        z = z+1 
        print("match number: " + str(z))
        #iterates through teams in matches
        for i in alliance:
            #assigning team number to team
            team = sb.get_match(match = nummatch, fields = [i]).get(i)
            #assigning epa values to dict
            dict = (sb.get_team_match(match = nummatch, team = team, fields = ["team", "epa", "auto_epa", "teleop_epa", "endgame_epa", "rp_1_epa", "rp_2_epa"]))
            #add win/loss to dict
            if sb.get_team_match(team = team, match = nummatch, fields = ["alliance"]) == sb.get_match(match = nummatch, fields = ["winner"]):
                dict.update({"win":1})
            else:
                dict.update({"win":0})
            #adding iteration dict to full dictionary
            dictionary.append(dict)

#changing dictionary to df
df = pd.DataFrame(dictionary)
print(df)

#assigning to csv file
df.to_csv('2023epastatistics.csv')

#assigning data to df from one event
# dictionary = []
# z = 0
# for x in (sb.get_matches(event = "2019cur")):
#     nummatch = x.get("key")
#     print(z)
#     z = z+1 
#     for i in alliance:
#         team = sb.get_match(match = nummatch, fields = [i]).get(i)
#         dict = (sb.get_team_match(match = nummatch, team = team)) # , fields = ["team", "epa", "auto_epa", "teleop_epa", "endgame_epa", "rp_1_epa", "rp_2_epa"]
#         dictionary.append(dict)

# df = pd.DataFrame(dictionary)
# print(df)


# convert info. from loop into dataframe
# for x in alliance:
#     dict = sb.get_team_match(match = nummatch, team = sb.get_match(match= nummatch, fields = [x]).get(x))
#     dictionary.append(dict)
# df = pd.DataFrame(dictionary)
# print(df)

#add win/loss to dictionary
# dict = sb.get_team_match(team = 56, match = "2019cur_qm1")

# if sb.get_team_match(team = 56, match = "2019cur_qm1", fields = ["alliance"]) == sb.get_match(match = "2019cur_qm1", fields = ["winner"]):
#     dict.update({"win":1})
# else:
#     dict.update({"win":0})


# print(dict)

