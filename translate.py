def translate_community_label(name):
    s = ""
    if name[4]=="1":
        s = s + "p = 0.95, "
    elif name[4]=="2":
        s = s + "p = 0.85, "
    elif name[4]=="3":
        s = s + "p = 0.75, "
    if name[5]=="1":
        s = s + "$\mu$ = 0.1, "
    elif name[5]=="2":
        s = s + "$\mu$ = 0.2, "
    elif name[5]=="3":
        s = s + "$\mu$ = 0.3, "
    if name[6:]=="1":
        s = s + "k = 1"
    elif name[6:]=="2":
        s = s + "k = 2"
    elif name[6:]=="5":
        s = s + "k = 5"        
    elif name[6:]=="0":
        s = s + "k = 10"
    elif name[6:]=="15":
        s = s + "k = 15"        
    return s

def translate_edge_correlated(name):
    s = ""
    if name[4]=="1":
        s = s + "p = 0.8, "
    elif name[4]=="2":
        s = s + "p = 0.7, "
    elif name[4]=="3":
        s = s + "p = 0.6, "
    if name[5]=="1":
        s = s + "$\mu$ = 0.1, "
    elif name[5]=="2":
        s = s + "$\mu$ = 0.2, "
    elif name[5]=="3":
        s = s + "$\mu$ = 0.3, "
    if name[6:]=="1":
        s = s + "k = 1"
    elif name[6:]=="2":
        s = s + "k = 2"
    elif name[6:]=="5":
        s = s + "k = 5"        
    elif name[6:]=="0":
        s = s + "k = 10"
    elif name[6:]=="15":
        s = s + "k = 15"        
    return s
