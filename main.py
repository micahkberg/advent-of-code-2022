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


def day4():
    assignments = read_input("day4.txt")
    complete_overlap_count = 0
    partial_overlap_count = 0
    for pair in assignments:
        if pair == "":
            break
        ranges = list(map(lambda i: i.split("-"), pair.split(",")))
        if int(ranges[0][0])<=int(ranges[1][0]) and int(ranges[0][1])>=int(ranges[1][1]):
            complete_overlap_count+=1
        elif int(ranges[1][0])<=int(ranges[0][0]) and int(ranges[1][1])>=int(ranges[0][1]):
            complete_overlap_count+=1

        elf1 = set(range(int(ranges[0][0]),int(ranges[0][1])+1))
        elf2 = set(range(int(ranges[1][0]),int(ranges[1][1])+1))
        if len(elf1.intersection(elf2)) > 0:
            partial_overlap_count+=1

    print(f"complete overlaps = {complete_overlap_count}")
    print(f"partial overlaps = {partial_overlap_count}")


def day5():
    info = read_input("day5.txt")
    instructions = []
    stacks = dict()

    for i in range(1,10):
        stacks[i] = []
    for row in info:
        if row.startswith("["):
            for i in range(1,10):
                col = 1+4*(i-1)
                if row[col] != ' ':
                    stacks[i]+=row[col]

        elif row.startswith("m"):
            instructions.append(row)
    print(stacks)

    total_boxes = 0
    for i in range(1,10):
        total_boxes+=len(stacks[i])
    print(f"total boxes before executing instructions: {total_boxes}")

    for instruction in instructions:
        move, amount, frm, home, to, destination = instruction.split(" ")
        moved_segment = stacks[int(home)][:int(amount)]
        moved_segment.reverse()
        stacks[int(destination)] = moved_segment+stacks[int(destination)]
        stacks[int(home)] = stacks[int(home)][int(amount):]

    total_boxes = 0
    for i in range(1, 10):
        total_boxes += len(stacks[i])
    print(f"total boxes after executing instructions: {total_boxes}")

    output_str = ""
    for i in range(1,10):
        output_str+=stacks[i][0]
    print(output_str)

    # try 1 BPCZJLFJW (which turned out to be part 2 becuase i didnt realize they werent reversed)

day5()