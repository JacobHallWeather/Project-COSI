import math
import numpy as np

# Define the percentiles
key={}
key["Manhattan"]={100:115000,99.9:106644,99.5:82000,99:74960,95:49000,90:33000,85:28000,80:23300,75:20000,70:17000,65:14460,60:12000,55:11000,50:9300,40:6900,30:5160,20:3940,10:2800,0:0}
key["Zuma"]={100:100000,99.9:100000,99.5:78500,99:70000,95:40000,90:30000,85:25000,80:20000,75:16000,70:12000,65:9000,60:6500,55:5000,50:4000,40:2500,30:2000,20:1500,10:1000,0:0}

# Turn a raw attendance number into a percentile
def raw2percent(a_list, where):
    keys = list(key[where].keys())
    keys = sorted(keys, reverse=True)

    percentile_list = []
    for a in a_list:
        if np.isnan(a):  # Check if input value is NaN
            percentile_list.append(np.nan)
        else:
            for i in range(0, len(keys)):
                p = keys[i]  # percentile
                v = key[where][p]  # attendance value
                if a > v:
                    percentile_list.append(p)
                    break
                if i == len(keys) - 1:
                    percentile_list.append(0)
    return percentile_list

def percent2index(percentile_list, where):
    keys = list(key[where].keys())
    keys = sorted(keys, reverse=True)
    num_keys = len(keys) - 1

    index_list = []
    for p in percentile_list:
        if np.isnan(p):  # Check if input value is NaN
            index_list.append(np.nan)
        else:
            for i in range(0, len(keys)):
                percentile = keys[i]  # percentile
                v = key[where][percentile]  # attendance value
                if p > percentile:
                    ind = len(keys) - i
                    index_list.append(round(10.0 * ind / num_keys)) #index ranges from 0 to 17
                    break
                if i == len(keys) - 1:
                    index_list.append(0)
    return index_list


# Turn the percentile into a range of raw attendance values by beach
def percent2raw(a_list, where):
    if where not in key:
        return [np.nan] * len(a_list)  # beach location not found
    keys = list(key[where].keys())
    keys = sorted(keys, reverse=True)

    raw_list = []
    for a in a_list:
        if np.isnan(a):  # Check if input value is NaN
            raw_list.append(np.nan)
        else:
            if isinstance(a, str):
                parts = a.split('-')
                if len(parts) == 2:
                    raw_list.append(a)
                else:
                    raw_list.append(np.nan)
            else:
                for i in range(0, len(keys)):
                    p = keys[i]  # percentile
                    v = key[where][p]  # attendance value
                    if a == p:
                        if i == 0:
                            raw_list.append(key[where][p])
                        else:
                            floor = v
                            ceiling = key[where][keys[i - 1]]
                            raw_list.append(str(floor) + "-" + str(ceiling))
                        break
                    if i == len(keys) - 1:
                        raw_list.append("0-" + str(key[where][keys[len(keys) - 1]]))
    return raw_list
