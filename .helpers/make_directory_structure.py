import os

for d in range(1,6):
    for slot in range(1,3):
        p=os.path.join("tutorials", "day"+str(d), "tutorial"+str(slot)) 
        if not os.path.isdir(p):
            os.makedirs(p)
        content="""Day {} Tutorial {} 
        ===================""".format(d,slot)
        with open(os.path.join(p, "readme.md"), "w") as f:
            f.write(content)
