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


day1()
