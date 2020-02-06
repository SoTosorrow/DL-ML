import numpy as np
a = np.array([(3,4),(3,6),(7,3),(4,7),(3,8),(8,5),(4,5),(4,1),(7,4),(5,5)])
lines = ""
for i in a:
    for j in a:
        dis = np.sqrt(np.sum((i-j)**2))
        lines+="%.2f"%dis+","
    lines+="\n"
file = open("result.csv",mode="w",encoding="utf-8")
file.write(lines)
file.close()
