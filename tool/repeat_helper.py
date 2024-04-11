from collections import defaultdict


def getsubs(loc, s):
    substr = s[loc:]
    i = -1
    while (len(substr) > 1):
        yield substr
        substr = s[loc:i]
        i -= 1

def Repeat(r: str):
    result = ''
    occ = defaultdict(int)
    for i in [" ", '#', '、', '.']:
        r = r.replace(i, '')
    # list all occurrences of all substrings
    for i in range(len(r)):
        for sub in getsubs(i, r):
            occ[sub] += 1

    if occ:
        filtered = sorted(occ.items(), key=lambda x: x[1], reverse=True)
        # filter out all sub strings with fewer than 2 occurrences
        filtered = [x for x in filtered if x[1] > 1]
        if filtered:
            maxkey = filtered[0][0] # Find longest string
            result = maxkey
    return result


if __name__ == '__main__':
    print(Repeat('# Ser # # Ser # # Ser # # Ser # # Ser # Ser #  # Ser # Ser # # Ser # Ser # Ser # Ser # Ser # Ser # Ser # Ser # Ser # Ser # Ser # Ser # Ser # Ser # Ser # Ser # Ser # Ser # Ser # Ser '))