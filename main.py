from operator import itemgetter
import numpy as np
import itertools

def read_input(fname):
    f = open(fname, "r")
    contents = f.read().strip().split("\n")
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


def day6():
    datastream = read_input("day6.txt")[0]
    for i in range(len(datastream)):
        current_signal=datastream[i:i+14]
        if len(set(current_signal))==14:
            print(i+14)
            break


def day7():

    class File:
        def __init__(self, name, size, parent):
            self.name = name
            self.size = size
            self.sub_files = []
            self.parent = parent

        def get_size(self):
            if self.size>0:
                return self.size
            else:
                size = 0
                for file in self.sub_files:
                    if type(file) == File:
                        print(file.name)
                        size+=file.get_size()
                return size
    commands = read_input("day7.txt")
    current_directory = File("None", 0, None)
    directories = []
    for line in commands:
        if line == "":
            break
        parts = line.split(" ")
        if line.startswith("$"):
            if parts[1] == "cd":
                if parts[2] == "..":
                    current_directory = current_directory.parent
                else:
                    found_match = False
                    for file in current_directory.sub_files:
                        if file.name == parts[2]:
                            current_directory = file
                            found_match = True
                    if not found_match:
                        current_directory = File(parts[2], 0 , current_directory)
                        directories.append(current_directory)
                        print("changing directory to a place that we didnt know about pre $ cd")
        else:
            if parts[1] not in list(map(lambda i: i.name, current_directory.sub_files)):
                if line.startswith("dir"):
                    new_dir = File(parts[1], 0, current_directory)
                    directories.append(new_dir)
                    current_directory.sub_files.append(new_dir)
                else:
                    current_directory.sub_files.append(File(parts[1], int(parts[0]), current_directory))
    small_ones_total = 0
    for directory in directories:
        if directory.name == "/":
            current_file_volume = directory.get_size()
        if directory.get_size() <= 100000:
            print(directory.name)
            small_ones_total += directory.get_size()
    print(small_ones_total)

    diskspace = 70000000
    need_free = 30000000
    dir_delete_min = need_free - (diskspace - current_file_volume)
    current_min = 70000000000
    for directory in directories:
        if directory.get_size() >= dir_delete_min and directory.get_size() < current_min:
            current_min = directory.get_size()
    print(current_min)

def day8():

    def check_vis(tree_val, list_of_other_trees):
        for other_tree in list_of_other_trees:
            if other_tree >= tree_val:
                return False
        return True

    lines = read_input("day8.txt")[:-1]
    visible_grid = [[False for _ in range(len(lines))] for _ in range(len(lines[0]))]
    for y in range(len(lines)):
        for x in range(len(lines[0])):
            tree_val = int(lines[y][x])
            L = list(map(int, list(lines[y][:x])))
            R = list(map(int, list(lines[y][x+1:])))
            U = list(map(lambda i: int(i[x]), lines[:y]))
            D = list(map(lambda i: int(i[x]), lines[y+1:]))
            for dir in L,R,U,D:
                dir.append(-1)

            if tree_val > max(L) or tree_val > max(R) or tree_val > max(U) or tree_val > max(D):
                visible_grid[y][x] = True
    vis_count = 0
    for row in visible_grid:
        for tree in row:
            if tree:
                vis_count += 1
    print(vis_count)

    # part 2
    # try 1 1947582
    # try 2 2042586 ?
    # try 3 291840

    def calc_score(x,y):
        tree_height = int(lines[y][x])
        score = 1
        L = list(map(int, list(lines[y][:x])))[::-1]
        R = list(map(int, list(lines[y][x + 1:])))
        U = list(map(lambda i: int(i[x]), lines[:y]))[::-1]
        D = list(map(lambda i: int(i[x]), lines[y + 1:]))
        for dir in L,R,U,D:
            dir_score = 0
            for dir_tree in dir:
                dir_score+=1
                if dir_tree >= tree_height:
                    break
            score *= dir_score
        return score


    best_score = 0
    for y in range(len(lines)):
        for x in range(len(lines[0])):
            tree_score = calc_score(x,y)
            best_score = tree_score if tree_score > best_score else best_score
    print(best_score)


def day9():
    h = (0, 0)
    t = (0, 0)
    visited = set()
    moves = read_input("day9.txt")
    dirs = {"R": (1,0), "U": (0,1), "L": (-1,0), "D": (0,-1)}
    for move in moves:
        if move == "":
            break
        direction, steps = move.split(" ")
        for step in range(int(steps)):
            h = (h[0]+dirs[direction][0], h[1]+dirs[direction][1])
            rope_vector = (h[0]-t[0],h[1]-t[1])
            if abs(rope_vector[0]) == 2 or abs(rope_vector[1]) == 2:
                tail_move = tuple(map(lambda i: 0 if rope_vector[i]==0 else rope_vector[i]/abs(rope_vector[i]), [0,1]))
                t = (t[0]+tail_move[0],t[1]+tail_move[1])
                visited.add(t)
    print(len(visited))
    visited = set()
    class Knot:
        def __init__(self):
            self.position = (0,0)

        def move(self, direction):
            self.position = (self.position[0] + direction[0], self.position[1] + direction[1])

    knots = [Knot() for _ in range(10)]

    for move in moves:
        if move == "":
            break
        direction, steps = move.split(" ")
        for step in range(int(steps)):
            knots[0].move(dirs[direction])
            for i in range(9):
                cur_knot = knots[i+1]
                last_knot = knots[i]
                rope_vector = (last_knot.position[0] - cur_knot.position[0], last_knot.position[1] - cur_knot.position[1])
                if abs(rope_vector[0]) == 2 or abs(rope_vector[1]) == 2:
                    tail_move = tuple(
                        map(lambda i: 0 if rope_vector[i] == 0 else rope_vector[i] / abs(rope_vector[i]), [0, 1]))
                    cur_knot.move(tail_move)
            visited.add(knots[9].position)


    print(len(visited))


def day10():
    commands = read_input("day10.txt")
    cycle = 0
    X = 1
    current_command = None
    wait = False
    signal_strengths = []
    image = ""
    while True:
        cycle+=1
        current_draw_pos = (cycle - 1)%40
        if X-1 <= current_draw_pos <= X+1 and cycle<=240:
            image+="#"
        elif cycle<=240:
            image+="."
        if cycle%40 == 20:
            signal_strength = X*cycle
            signal_strengths.append(signal_strength)
        if cycle%40 == 0:
            image+="\n"
        if not current_command:
            current_command = commands.pop(0)
            if current_command.startswith("addx"):
                wait = True
        if current_command == "noop":
            current_command = None
        elif current_command.startswith("addx"):
            if not wait:
                X += int(current_command.split(" ")[1])
                current_command = None
            else:
                wait = False

        if len(commands) == 0 and not wait:
            break
    print(sum(signal_strengths[0:6]))
    print(image)
    # part 1 try 1: 14680 (too high)
    # part 2 ZUPRFECL

def day11():
    f = open("day11.txt", "r")
    contents = f.read().split("\n\n")
    f.close()
    monkey_texts = list(map(lambda i: i.split("\n"), contents))

    part_2 = True

    class Monkey:
        def __init__(self, input_lists):
            self.items = list(map(int, input_lists[1].strip().strip("Starting items: ").split(", ")))
            self.operations = input_lists[2].strip().split(" ")[-2:]
            self.test_divisor = int(input_lists[3].split(" ")[-1])
            self.true_destination = int(input_lists[4].split(" ")[-1])
            self.false_destination = int(input_lists[5].split(" ")[-1])
            self.inspections_total = 0
            self.other_monkeys = []
            self.part_2_divisor = 1

        def inspect(self):
            self.inspections_total += 1
            looking_at = self.items.pop(0)
            if self.operations[0] == '+':
                looking_at += int(self.operations[1])
            elif self.operations[0] == '*':
                looking_at = looking_at * int(self.operations[1]) if self.operations[1] != "old" else looking_at*looking_at
            if not part_2:
                looking_at = looking_at//3
            else:
                looking_at = looking_at%part_2_divisor
            return looking_at

        def check_item(self, item):
            if item%self.test_divisor == 0:
                return self.true_destination
            else:
                return self.false_destination

        def take_turn(self):
            while len(self.items) > 0:
                throwing = self.inspect()
                destination = self.check_item(throwing)
                self.other_monkeys[destination].items.append(throwing)


    monkeys = []
    for monkey_text in monkey_texts:
        monkeys.append(Monkey(monkey_text))


    part_2_divisor = 1
    for monkey in monkeys:
        monkey.other_monkeys = monkeys
        part_2_divisor *= monkey.test_divisor

    for monkey in monkeys:
        monkey.part_2_divisor = part_2_divisor

    for round in range(20 if not part_2 else 10000):
        for monkey in monkeys:
            monkey.take_turn()

    top_monkeys = sorted(list(map(lambda i: i.inspections_total, monkeys)), reverse=True)
    monkey_business = top_monkeys[0] * top_monkeys[1]
    print(monkey_business)
    # part 1 try 1: 55224 too low
    # part 2 try 2 58322

def day12():
    heightmap = read_input("day12.txt")
    def get_height(x,y):
        if heightmap[y][x] == "S":
            return "a"
        elif heightmap[y][x] == "E":
            return "z"
        else:
            return heightmap[y][x]

    path_map = {}
    for y in range(len(heightmap)):
        for x in range(len(heightmap[0])):
            if heightmap[y][x] == "S":
                start = (x,y)
            elif heightmap[y][x] == "E":
                end = (x,y)
            path_map[(x,y)] = []
            for direction in [[0,1],[1,0],[-1,0],[0,-1]]:
                if 0<=x+direction[0]<len(heightmap[0]) and 0<=y+direction[1]<len(heightmap):
                    if ord(get_height(x,y))+1 >= ord(get_height(x+direction[0],y+direction[1])):
                        path_map[(x,y)].append((x+direction[0],y+direction[1]))
    distances = {start: 0}
    queue = [start]
    seen = set()
    counter = 0
    while len(queue)>0:
        current = queue.pop(0)
        seen.add(current)
        for adj_cell in path_map[current]:
            if adj_cell in distances.keys():
                distances[adj_cell] = min([distances[adj_cell],distances[current]+1])
            else:
                distances[adj_cell] = distances[current] + 1
                queue.append(adj_cell)
            if current == end:
                print(distances[end])
    print(distances[end])
    print(f"{len(seen)}/{len(heightmap) * len(heightmap[0])}")

def day12part2():
    heightmap = read_input("day12.txt")

    def get_height(x, y):
        if heightmap[y][x] == "S":
            return "a"
        elif heightmap[y][x] == "E":
            return "z"
        else:
            return heightmap[y][x]

    path_map = {}
    for y in range(len(heightmap)):
        for x in range(len(heightmap[0])):
            if heightmap[y][x] == "E":
                end = (x,y)
            path_map[(x,y)] = []
            for direction in [[0,1],[1,0],[-1,0],[0,-1]]:
                if 0<=x+direction[0]<len(heightmap[0]) and 0<=y+direction[1]<len(heightmap):
                    if ord(get_height(x, y)) <= ord(get_height(x+direction[0],y+direction[1]))+1:
                        path_map[(x,y)].append((x+direction[0],y+direction[1]))
    distances = {end: 0}
    queue = [end]
    dist = 999999
    seen = set()
    while len(queue) > 0:
        current = queue.pop(0)
        seen.add(current)
        for adj_cell in path_map[current]:
            current_height = get_height(current[0],current[1])
            if adj_cell in distances.keys():
                if distances[adj_cell] > distances[current] + 1:
                    distances[adj_cell] = distances[current] + 1
                    queue.append(adj_cell)
            else:
                distances[adj_cell] = distances[current] + 1
                queue.append(adj_cell)
        if get_height(current[0],current[1]) == "a" and distances[current] < dist:
            dist = distances[current]
            print(dist)
    print(f"{len(seen)}/{len(heightmap)*len(heightmap[0])}")

def day13():
    data = read_input("day13.txt")
    pairs = []
    pair = []
    for packet in data:
        if len(pair)<2 and packet != "":
            pair.append(eval(packet))
        elif len(pair)==2:
            pairs.append(pair)
            pair = []
    pairs.append(pair)
    index_sum = 0
    index = 0


    def test(a,b):
        if type(a)==int and type(b)==int:
            if a<b:
                return "PASS"
            elif a==b:
                return "CONTINUE"
            else:
                return "FAIL"
        elif type(a) == list and type(b) == list:
            comparing = True
            i=0
            while comparing:
                if i==len(a) and i==len(b):
                    return "CONTINUE"
                elif i==len(a) and i<len(b):
                    return "PASS"
                elif i<len(a) and i==len(b):
                    return "FAIL"
                else:
                    subresult = test(a[i],b[i])
                    if subresult == "CONTINUE":
                        i+=1
                    else:
                        return subresult

        else:
            a = [a] if type(a) == int else a
            b = [b] if type(b) == int else b
            return test(a, b)

    organized_packets = []
    for pair in pairs:
        index+=1
        if test(pair[0],pair[1]) == "PASS":
            index_sum += index
            organized_packets.append(pair[0])
            organized_packets.append(pair[1])
        else:
            organized_packets.append(pair[1])
            organized_packets.append(pair[0])

    organized_packets.append([[2]])
    organized_packets.append([[6]])

    still_solving = True
    while still_solving:
        still_solving = False
        for i in range(len(organized_packets)-1):
            if test(organized_packets[i],organized_packets[i+1]) == "FAIL":
                test_len = len(organized_packets)
                a = organized_packets[i]
                b = organized_packets[i+1]
                organized_packets[i+1] = a.copy()
                organized_packets[i] = b.copy()
                if test_len != len(organized_packets):
                    print("packet recomposition broken")
                still_solving = True
    decoder_key = 1
    for i in range(len(organized_packets)-1):
        if test(organized_packets[i],organized_packets[i+1]) == "FAIL":
            print("Didnt sort")
        elif organized_packets[i] == [[2]] or organized_packets[i] == [[6]]:
            decoder_key*=i+1
    print(decoder_key)


    print(f"{index_sum}/{sum(list(range(1,len(pairs)+1)))}")

    # part 1 try 1 594
    # part 1 try 2 334 too low
    # part 1 try 3 5693 too low


def day14():
    rock_paths = read_input("day14.txt")
    part2 = True
    # draw rocks:
    rock_positions = set()
    max_x = 500
    min_x = 500
    max_y = 50
    min_y = 10

    for path in rock_paths:
        path_steps = list(map(lambda i: i.split(","), path.split(" -> ")))
        # too lazy to figure out the nested map to make these all tupled ints
        for i in range(len(path_steps)):
            path_steps[i] = tuple(map(int, path_steps[i]))

        for step_n in range(len(path_steps)-1):
            current_step = path_steps[step_n]
            next_step = path_steps[step_n+1]
            rock_positions.add(next_step)
            travel_vector = (next_step[0]-current_step[0],next_step[1]-current_step[1])
            for i in range(0,sum(travel_vector),sum(travel_vector)//abs(sum(travel_vector))):
                if travel_vector[0]==0:
                    new_coord = (current_step[0],current_step[1]+i)
                else:
                    new_coord = (current_step[0]+i,current_step[1])
                rock_positions.add(new_coord)
                if new_coord[0] > max_x:
                    max_x = new_coord[0]
                elif new_coord[0] < min_x:
                    min_x = new_coord[0]
                if new_coord[1] < max_y:
                    max_y = new_coord[1]
                elif new_coord[1] > min_y:
                    min_y = new_coord[1]




    sand_positions = set()
    Sand_pouring = True

    def can_move(pos):
        if pos[1]==min_y+1 and part2:
            return False
        elif (pos[0],pos[1]+1) not in rock_positions and (pos[0],pos[1]+1) not in sand_positions:
            return (pos[0], pos[1]+1)
        elif (pos[0] -1 ,pos[1]+1) not in rock_positions and (pos[0] - 1 ,pos[1]+1) not in sand_positions:
            return (pos[0] - 1 , pos[1] + 1)
        elif (pos[0] + 1 ,pos[1]+1) not in rock_positions and (pos[0] + 1 ,pos[1]+1) not in sand_positions:
            return (pos[0] + 1 , pos[1] + 1)
        else:
            return False

    # draw sand castle

    def view(cur_sand = None):
        if not cur_sand:
            y1 = max_y-15
            y2 = min_y+5
            x1 = min_x-5
            x2 = max_x+5
        else:
            y1 = cur_sand[1]-5
            y2 = cur_sand[1]+5
            x1 = cur_sand[0]-5
            x2 = cur_sand[0]+5
        for y in range(y1,y2):
            line = ""
            for x in range(x1,x2):
                if (x,y) in rock_positions:
                    line+="#"
                elif (x,y) in sand_positions:
                    line+="O"
                elif (x,y) == cur_sand:
                    line+="~"
                else:
                    line+="."
            print(line)
        print("\n")

    while Sand_pouring:
        sand_position = (500,0)
        in_abyss = False
        while can_move(sand_position):
            sand_position = can_move(sand_position)
            #view(sand_position)
            if sand_position[1]>min_y and not part2:
                in_abyss = True
                break
        if not in_abyss:
            sand_positions.add(sand_position)
        elif in_abyss:
            break
        elif (500,0) in sand_positions:
            break

        if len(sand_positions)%100 == 0:
            #view()
            print(len(sand_positions))
            pass
    print(len(sand_positions))
    # part 1 try 1 595, too low



def day14_less_brute_force():

    # i think maybe tracking all the sand falling so much is probably too slow,
    # we can just paint down since we know what sand will look like when its not moving (i did watch my view() a bunch)

    rock_paths = read_input("day14.txt")

    # draw rocks:
    rock_positions = set()
    sand_positions = {(500,0)}
    max_x = 500
    min_x = 500
    max_y = 50
    min_y = 10

    for path in rock_paths:
        path_steps = list(map(lambda i: i.split(","), path.split(" -> ")))
        # too lazy to figure out the nested map to make these all tupled ints
        for i in range(len(path_steps)):
            path_steps[i] = tuple(map(int, path_steps[i]))

        for step_n in range(len(path_steps) - 1):
            current_step = path_steps[step_n]
            next_step = path_steps[step_n + 1]
            rock_positions.add(next_step)
            travel_vector = (next_step[0] - current_step[0], next_step[1] - current_step[1])
            for i in range(0, sum(travel_vector), sum(travel_vector) // abs(sum(travel_vector))):
                if travel_vector[0] == 0:
                    new_coord = (current_step[0], current_step[1] + i)
                else:
                    new_coord = (current_step[0] + i, current_step[1])
                rock_positions.add(new_coord)
                if new_coord[0] > max_x:
                    max_x = new_coord[0]
                elif new_coord[0] < min_x:
                    min_x = new_coord[0]
                if new_coord[1] < max_y:
                    max_y = new_coord[1]
                elif new_coord[1] > min_y:
                    min_y = new_coord[1]


    def can_move(pos):
        if pos[1] == min_y + 1:
            return False
        elif (pos[0], pos[1] + 1) not in rock_positions and (pos[0], pos[1] + 1) not in sand_positions:
            return (pos[0], pos[1] + 1)
        elif (pos[0] - 1, pos[1] + 1) not in rock_positions and (pos[0] - 1, pos[1] + 1) not in sand_positions:
            return (pos[0] - 1, pos[1] + 1)
        elif (pos[0] + 1, pos[1] + 1) not in rock_positions and (pos[0] + 1, pos[1] + 1) not in sand_positions:
            return (pos[0] + 1, pos[1] + 1)
        else:
            return False

    def view(cur_sand=None):
        if not cur_sand:
            y1 = -5
            y2 = min_y + 5
            x1 = min_x - 5
            x2 = max_x + 5
        else:
            y1 = cur_sand[1] - 5
            y2 = cur_sand[1] + 5
            x1 = cur_sand[0] - 5
            x2 = cur_sand[0] + 5
        for y in range(y1, y2):
            line = ""
            for x in range(x1, x2):
                if (x, y) in rock_positions:
                    line += "#"
                elif (x, y) in sand_positions:
                    line += "O"
                elif (x, y) == cur_sand:
                    line += "~"
                else:
                    line += "."
            print(line)
        print("\n")

    new_sands = {(500, 0)}
    #view()
    while len(new_sands) > 0:
        cur_sand = new_sands.pop()
        while can_move(cur_sand):
            new_sand = can_move(cur_sand)
            new_sands.add(new_sand)
            sand_positions.add(new_sand)
        if len(sand_positions) % 100 == 0:
            # view()
            print(f"to simulate: {len(new_sands)}. total sand: {len(sand_positions)}")
            pass
       #view()

    view()
    print(len(sand_positions))

    # part 1 try 1 595, too low
    # part 2 try 1 24590, too high




def day15():
    sensors = read_input("day15.txt")
    cant_be = 0
    target_row = 2000000
    on_target_row = set()
    ranges = []

    class Zone:
        def __init__(self, circle_dict):
            self.center = circle_dict["center"]
            self.radius = circle_dict["radius"]

    def dist(a,b):
        if type(a)==Zone and type(b)==Zone:
            return abs(a.center[0]-b.center[0]) + abs(a.center[1]-b.center[1])
        elif type(a)==tuple and type(b)==tuple:
            return abs(a[0]-b[0]) + abs(a[1]-b[1])

    def intersect(a,b):
        if dist(a,b) < a.radius+b.radius:
            return True
        else:
            return False

    def adjacent(a,b):
        if dist(a,b) == a.radius+b.radius:
            return True
        else:
            return False

    for sensor in sensors:
        parts = sensor.split(" ")
        sensor_pos = (int(parts[2].strip("x=,")),int(parts[3].strip("y=:")))
        beacon_pos = (int(parts[8].strip("x=,")),int(parts[9].strip("y=")))
        man_dist = abs(sensor_pos[0]-beacon_pos[0])+abs(sensor_pos[1]-beacon_pos[1])
        if abs(sensor_pos[1]-target_row)<=man_dist:
            excess = man_dist - abs(sensor_pos[1]-target_row)
            new_range = (sensor_pos[0]-excess,sensor_pos[0]+excess)
            ranges.append(new_range)
        if beacon_pos[1]==target_row:
            on_target_row.add(beacon_pos[0])
    ranges.sort()
    for i in range(len(ranges)):
        for j in range(i+1,len(ranges)):
            if ranges[i][1]>=ranges[j][0]:
                ranges[j] = (ranges[i][1]+1, ranges[j][1])
        if ranges[i][0]<=ranges[i][1]:
            cant_be+= ranges[i][1]-ranges[i][0]+1
    cant_be -= len(on_target_row)
    print(f"part 1: {cant_be}")
    # part 1 try 1 4999656 too low, double counting beacons and not correctly evaluating overlapping segments
    # part 1 try 2 5374697 too high, forgot to take off test row value
    # part 1 try 3 5142231

    # going to try and search along the perimeters of each zone
    #  circles = []
    # for sensor in sensors:
    #   parts = sensor.split(" ")
    #    sensor_pos = (int(parts[2].strip("x=,")),int(parts[3].strip("y=:")))
    #    beacon_pos = (int(parts[8].strip("x=,")),int(parts[9].strip("y=")))
    #    man_dist = abs(sensor_pos[0] - beacon_pos[0]) + abs(sensor_pos[1] - beacon_pos[1])
    #    new_circle = {"center": sensor_pos, "radius": man_dist}
    #    circles.append(Zone(new_circle))

    #for quadruple in itertools.combinations(circles, 4):
    #    pairings = []
    #    for pair in itertools.combinations(quadruple, 2):
    #        if dist(pair[0],pair[1]) == pair[0].radius + pair[1].radius+1:
    #            pairings.append(pair)
    #    if pairings==2:
    #        print(quadruple)


    # iterating thru each row, a lot faster than i thought lol
    data = []
    for sensor in sensors:
        parts = sensor.split(" ")
        sensor_pos = (int(parts[2].strip("x=,")), int(parts[3].strip("y=:")))
        beacon_pos = (int(parts[8].strip("x=,")), int(parts[9].strip("y=")))
        man_dist = abs(sensor_pos[0] - beacon_pos[0]) + abs(sensor_pos[1] - beacon_pos[1])
        data.append({"sensor": sensor_pos, "man_dist":man_dist })

    for y in range(4000000):
        x = 0
        ranges = []
        for sensor in data:
            sensor_pos = sensor["sensor"]
            man_dist = sensor["man_dist"]
            if abs(sensor_pos[1] - y) <= man_dist:
                excess = man_dist - abs(sensor_pos[1] - y)
                new_range = (sensor_pos[0] - excess, sensor_pos[0] + excess)
                ranges.append(new_range)
        ranges.sort()

        while x <= 4000000:
            next_range = ranges.pop(0)
            if next_range[0]<x<next_range[1]:
                x = next_range[1]+1
            elif next_range[0]==x+1:
                print(f"({x},{y})")
                print(x*4000000+y)
        if y%100000 == 0:
            print(f"{100*y/4000000}")

    # day 15 part 2 try 1 16000002765172 too high
    #               try 2 10884459367718 !!!




day15()

def day16():
    tunnel_info = read_input("day16.txt")
    tunnels = dict()

    for line in tunnel_info:
        new_tunnel = dict()
        new_tunnel["name"] = line.split(" ")[1]
        new_tunnel["rate"] = int(line.split(" ")[4].strip("rate=;"))
        new_tunnel["connections"] = list(map(lambda i: i.strip(","), line.split(" ")[9:]))
        tunnels[new_tunnel["name"]] = new_tunnel
    important_valves = []
    for tunnel in tunnels.keys():
        if tunnels[tunnel]["rate"] > 0:
            important_valves.append(tunnel)
    print(important_valves)
    start = "AA"
    next_list = {"AA"}
    distances = {"AA":0}
    while len(next_list) > 0:
        current_valve = next_list.pop(0)


