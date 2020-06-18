import cassiopeia as cass
from cassiopeia.data import Queue, Position, Region
from cassiopeia.core import Summoner, MatchHistory, Match
from cassiopeia import Queue, Patch
import pandas as pd
from pandas import DataFrame as df
import numpy as np
import random, arrow, csv, time, pdb
import datetime

my_api_key = "RGAPI-ab78bfa8-2586-41b6-9a7a-cdf190938a67"
queueType = Queue.ranked_solo_fives
patch_ver = "9.19"
matchdata_num_of_each_region = 40000
test_sample_per_region = 2500
save_csv_per = 8000

# ID from the match data (which it provides directly).
champion_id_to_name_mapping = {champion.id: champion.name for champion in cass.get_champions(region="NA")}

def save_gosu_player(region : str):
    lol_gosu_names = []
    try:
        challenger_league = cass.get_challenger_league(queue=queueType, region=region)
        challenger_entries = challenger_league.entries

        for challenger in challenger_entries:
            lol_gosu_names.append(challenger.summoner.account_id)
        
        grandmaster_league = cass.get_grandmaster_league(queue=queueType, region=region)
        grandmaster_entries = grandmaster_league.entries

        for grandmaster in grandmaster_entries:
            lol_gosu_names.append(grandmaster.summoner.account_id)

        master_league = cass.get_master_league(queue=queueType, region=region)
        master_entries = master_league.entries
        for master in master_entries:
            lol_gosu_names.append(master.summoner.account_id)
    except:
        print("Server error raise. Wait for 1 second.")
        time.sleep(1)
        pass

    return lol_gosu_names

def collect_gosu_ids(region_str):
    worlds_gosu_players_id = {}
    
    try:
        with open("./LOLData/gosuIDs"+patch_ver+".csv","r") as csvfile:
            csvreader = csv.reader(csvfile)
            for region_name, gosu_ids in  zip(region_str, csvreader):
                worlds_gosu_players_id[region_name] = gosu_ids
            
    except:
        for region_name in region_str:
            gosu_players = save_gosu_player(region_name)
            worlds_gosu_players_id[region_name] = gosu_players
        
        with open("./LOLData/gosuIDs"+patch_ver+".csv","w", newline="") as csvfile:
            for gosu_ids in worlds_gosu_players_id.values():
                csvwriter = csv.writer(csvfile)
                csvwriter.writerow(gosu_ids)

    return worlds_gosu_players_id

def filter_match_history(summoner, patch):
    end_time = patch.end
    if end_time is None:
        end_time = arrow.now()
    match_history = MatchHistory(summoner=summoner, queues={queueType}, begin_time=patch.start, end_time=end_time)
    return match_history


def collect_gosu_matches(players_id):
    match_ids_dict = {}

    try:
        region_str = players_id.keys()
        with open("./LOLData/matchIDs"+patch_ver+".csv","r") as csvfile:
            csvreader = csv.reader(csvfile)
            for region_name, gosu_ids in  zip(region_str, csvreader):
                match_ids_dict[region_name] = gosu_ids

    except:
        for region_name in players_id.keys():
            patch = Patch.from_str(patch_ver, region=region_name)

            summoner_ids = players_id[region_name]

        
            match_ids = set([])
            for i, summoner_id in enumerate(summoner_ids):
                try:
                    new_summoner = Summoner(account_id=summoner_id, region=region_name)
                    matches = filter_match_history(new_summoner, patch)
                    match_ids.update([match.id for match in matches])
                    print('Now match ids length is {}'.format(len(match_ids)))
                    print('Now used summoners are {} / {}'.format(i+1,len(summoner_ids)))

                except:
                    print("Server error raise. Wait for 1 second.")
                    time.sleep(1)
                    pass

            match_ids = list(match_ids)
            match_ids_dict[region_name] = match_ids
            with open("./LOLData/matchIDs"+patch_ver+".csv","a", newline="") as csvfile:
                csvwriter = csv.writer(csvfile)
                csvwriter.writerow(match_ids)

                # The above lines will trigger the match to load its data by iterating over all the participants.
                # If you have a database in your datapipeline, the match will automatically be stored in it.            
    return match_ids_dict

def collect_champ_comp(matches_id, seperation):
    #match_data={'blue_win':[],'blue_top_lane':[],'blue_jungle':[],'blue_mid_lane':[],'blue_bot_lane':[],'blue_bot_lane1':[],'blue_bot_lane2':[],
                #'red_top_lane':[],'red_jungle':[],'red_mid_lane':[],'red_bot_lane':[],'red_bot_lane1':[],'red_bot_lane2':[]}
    match_data={'blue_win':[],'blue_1':[],'blue_2':[],'blue_3':[],'blue_4':[],'blue_5':[],
                'red_1':[],'red_2':[],'red_3':[],'red_4':[],'red_5':[],'blue_death_diff':[],'red_death_diff':[], 'gold_diff':[], 'xp_diff':[], 'timeline':[]}

    for region_name in matches_id.keys():
        match_ids = matches_id[region_name]

        for k, match_id in enumerate(match_ids):
            try:
                match_id = int(match_id)
                new_match = Match(id=match_id, region=region_name)
            
                #if new_match.blue_team.participants[0].lane != None:
                frame_len = len(new_match.timeline.frames)
                match_data['timeline'].extend(list(range(frame_len)))
                
                blue_gold = [0 for col in range(frame_len)]
                red_gold = [0 for col in range(frame_len)]
                blue_xp = [0 for col in range(frame_len)]
                red_xp = [0 for col in range(frame_len)]
                blue_death = [0 for col in range(frame_len)]
                red_death = [0 for col in range(frame_len)]

                if new_match.blue_team.win:
                    for _ in range(frame_len):
                        match_data['blue_win'].append(1)
                else:
                    for _ in range(frame_len):
                        match_data['blue_win'].append(0)
                for i, blue_participant in enumerate(new_match.blue_team.participants):
                    for j, frame in enumerate(blue_participant.timeline.frames):
                        match_data['blue_'+str(i+1)].append(champion_id_to_name_mapping[blue_participant.champion.id])
                        blue_gold[j] += frame.gold_earned
                        blue_xp[j] += frame.experience
                        blue_death[j] += blue_participant.cumulative_timeline[datetime.timedelta(minutes=j)].deaths                        
                
                for i, red_participant in enumerate(new_match.red_team.participants):
                    for j, frame in enumerate(red_participant.timeline.frames):
                        match_data['red_'+str(i+1)].append(champion_id_to_name_mapping[red_participant.champion.id])
                        red_gold[j] += frame.gold_earned
                        red_xp[j] += frame.experience
                        red_death[j] += red_participant.cumulative_timeline[datetime.timedelta(minutes=j)].deaths
                
                for i in range(len(blue_death)):
                    if i == 0:
                        match_data['blue_death_diff'].append(0)
                    else:
                        match_data['blue_death_diff'].append(blue_death[i] - blue_death [i-1])
                for i in range(len(red_death)):
                    if i == 0:
                        match_data['red_death_diff'].append(0)
                    else:
                        match_data['red_death_diff'].append(red_death[i] - red_death [i-1])
                
                match_data['gold_diff'].extend((np.array(blue_gold) - np.array(red_gold)).tolist())
                match_data['xp_diff'].extend((np.array(blue_xp) - np.array(red_xp)).tolist())
                                
            except:
                print("Server error raise. Wait for 2 second.")
                time.sleep(2)
                pass
            len_list=[len(length) for length in match_data.values()]
            min_len = min(len_list)
            for data_list in match_data.values():
                if min_len < len(data_list):
                    for _ in range(frame_len):
                        del data_list[-1]

            print('Now match ID length is {} / {}'.format(k, len(match_ids)))
            print('Now match Data length is', len_list)

            if (k + 1) % save_csv_per == 0 or (k + 1) == len(match_ids):
                match_data_df = df(match_data)
                match_data_df.to_csv("./LOLData/MatchData"+patch_ver+seperation+".csv",mode='a')
                match_data.clear()
                #pdb.set_trace()
                match_data={'blue_win':[],'blue_1':[],'blue_2':[],'blue_3':[],'blue_4':[],'blue_5':[],
                'red_1':[],'red_2':[],'red_3':[],'red_4':[],'red_5':[],'blue_death_diff':[],'red_death_diff':[], 'gold_diff':[], 'xp_diff':[], 'timeline':[]}
    match_data.clear()

if __name__ == "__main__":
    cass.set_riot_api_key(my_api_key)
    '''
    region_str = []
    
    for i in Region:
        region_str.append(i.value)
    '''
    region_str = ["KR","EUW","NA","EUNE"]
    worlds_gosu_players_id_dict = collect_gosu_ids(region_str)

    match_ids_dict = collect_gosu_matches(worlds_gosu_players_id_dict)

    train_match_ids_dict = {}
    test_match_ids_dict = {}
    for region in region_str:
        test_match_ids_dict[region] = random.sample(match_ids_dict[region], test_sample_per_region)
        train_match_ids_dict[region] = list(set(match_ids_dict[region])-set(test_match_ids_dict[region]))
    del match_ids_dict

    collect_champ_comp(train_match_ids_dict, 'train')
    collect_champ_comp(test_match_ids_dict, 'test')



