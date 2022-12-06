from operator import itemgetter


def read_input(fname):
    f = open(fname, "r")
    contents = f.read().split("\n")
    f.close()
    return contents


def day1():
    # calorie counting
    cal_list = read_input("day1.txt")

    elf_array = []
    new_elf = []
    cals_list = []
    for item in cal_list:
        if item:
            new_elf.append(item)
        else:
            elf_array.append(new_elf)
            cals = sum(list(map(int, new_elf)))
            cals_list.append(cals)
            new_elf = []
    elf_array_sorted = sorted(cals_list)
    print(f"maximum number of calories is {elf_array_sorted[-1]} calories")
    print(f"the top 3 elves are carrying {sum(elf_array_sorted[-3:])}")
    print(elf_array_sorted[-3:])


def day2():
    rounds_raw = read_input("day2.txt")
    rounds = []
    for round in rounds_raw:
        if round:
            rounds.append(round.split(" "))
    # ax rock
    # by paper
    # cz scissors
    throw_score_dict = {"X": 1, "A": 1, "Y": 2, "B":2, "Z": 3, "C":3}
    winning = {"X": "C","Y": "A","Z": "B"}
    equals = {"X":"A", "Y":"B","Z":"C"}
    score = 0
    for round in rounds:
        if not round:
            break
        score += throw_score_dict[round[1]]
        if round[0] == winning[round[1]]:
            score+=6
        elif round[0] == equals[round[1]]:
            score+=3
    print(f"1. if XYZ is the throw: {score}")
    cypher = ["A","B","C"]
    strategy = {"X": 2,"Y": 0, "Z": 1}
    scorept2 = {"X": 0, "Y": 3, "Z":6}
    score = 0
    for round in rounds:
        score += scorept2[round[1]]
        score += throw_score_dict[cypher[(cypher.index(round[0])+strategy[round[1]])%3]]
    print(f"2. if XYZ is the strategy: {score}")

def day3():
    def score(character):
        if ord(character) >= 97:
            return ord(character)-96
        else:
            return ord(character)-64+26

    sacks = read_input("day3.txt")
    reorgs = ""
    for sack in sacks:
        bag1 = sack[len(sack)//2:]
        bag2 = sack[:len(sack)//2]
        for char in bag1:
            if char in bag2:
                reorgs+=char
                break
    badges = ""
    for groupnum in range(0,len(sacks)-3,3):
        e1,e2,e3 = set(sacks[groupnum]), set(sacks[groupnum+1]), set(sacks[groupnum+2])
        badges += e1.intersection(e2).intersection(e3).pop()
    print(sum(map(lambda i: score(i), reorgs)))
    print(sum(map(lambda i: score(i), badges)))



def day6():
    datastream = read_input("day6.txt")[0]
    for i in range(len(datastream)):
        current_signal=datastream[i:i+14]
        if len(set(current_signal))==14:
            print(i+14)
            break


day6()
