import os

gtav = os.listdir('/workspace/ssd0/byeongcheol/DGSS/Data/GTA5/labels/valid')

with open('gtav_split_val.txt', 'w') as f:
    for i in gtav:
        f.write(i + '\n')
