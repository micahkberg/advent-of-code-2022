import numpy as np

"""
Many apologies to myself in the future for deciding to put these all in the same .py
"""


def read_input(fname, strip=True):
    f = open("./inputs/"+fname, "r")
    if strip:
        contents = f.read().strip().split("\n")
    else:
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
    for round_line in rounds_raw:
        if round_line:
            rounds.append(round_line.split(" "))
    # ax rock
    # by paper
    # cz scissors
    throw_score_dict = {"X": 1, "A": 1, "Y": 2, "B": 2, "Z": 3, "C": 3}
    winning = {"X": "C", "Y": "A", "Z": "B"}
    equals = {"X": "A", "Y": "B", "Z": "C"}
    score = 0
    for round_line in rounds:
        if not round_line:
            break
        score += throw_score_dict[round_line[1]]
        if round_line[0] == winning[round_line[1]]:
            score += 6
        elif round_line[0] == equals[round_line[1]]:
            score += 3
    print(f"1. if XYZ is the throw: {score}")
    cypher = ["A", "B", "C"]
    strategy = {"X": 2, "Y": 0, "Z": 1}
    score_pt2 = {"X": 0, "Y": 3, "Z": 6}
    score = 0
    for round_line in rounds:
        score += score_pt2[round_line[1]]
        score += throw_score_dict[cypher[(cypher.index(round_line[0]) + strategy[round_line[1]]) % 3]]
    print(f"2. if XYZ is the strategy: {score}")


def day3():
    def score(character):
        if ord(character) >= 97:
            return ord(character) - 96
        else:
            return ord(character) - 64 + 26

    sacks = read_input("day3.txt")
    reorgs = ""
    for sack in sacks:
        bag1 = sack[len(sack) // 2:]
        bag2 = sack[:len(sack) // 2]
        for char in bag1:
            if char in bag2:
                reorgs += char
                break
    badges = ""
    for group_num in range(0, len(sacks) - 3, 3):
        e1, e2, e3 = set(sacks[group_num]), set(sacks[group_num + 1]), set(sacks[group_num + 2])
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
        if int(ranges[0][0]) <= int(ranges[1][0]) and int(ranges[0][1]) >= int(ranges[1][1]):
            complete_overlap_count += 1
        elif int(ranges[1][0]) <= int(ranges[0][0]) and int(ranges[1][1]) >= int(ranges[0][1]):
            complete_overlap_count += 1

        elf1 = set(range(int(ranges[0][0]), int(ranges[0][1]) + 1))
        elf2 = set(range(int(ranges[1][0]), int(ranges[1][1]) + 1))
        if len(elf1.intersection(elf2)) > 0:
            partial_overlap_count += 1

    print(f"complete overlaps = {complete_overlap_count}")
    print(f"partial overlaps = {partial_overlap_count}")


def day5():
    info = read_input("day5.txt")
    instructions = []
    stacks = dict()

    for i in range(1, 10):
        stacks[i] = []
    for row in info:
        if row.startswith("["):
            for i in range(1, 10):
                col = 1 + 4 * (i - 1)
                if row[col] != ' ':
                    stacks[i] += row[col]

        elif row.startswith("m"):
            instructions.append(row)
    print(stacks)

    total_boxes = 0
    for i in range(1, 10):
        total_boxes += len(stacks[i])
    print(f"total boxes before executing instructions: {total_boxes}")

    for instruction in instructions:
        move, amount, frm, home, to, destination = instruction.split(" ")
        moved_segment = stacks[int(home)][:int(amount)]
        moved_segment.reverse()
        stacks[int(destination)] = moved_segment + stacks[int(destination)]
        stacks[int(home)] = stacks[int(home)][int(amount):]

    total_boxes = 0
    for i in range(1, 10):
        total_boxes += len(stacks[i])
    print(f"total boxes after executing instructions: {total_boxes}")

    output_str = ""
    for i in range(1, 10):
        output_str += stacks[i][0]
    print(output_str)

    # try 1 "BPCZJLFJW" (which turned out to be part 2 because i didn't realize they weren't reversed)


def day6():
    datastream = read_input("day6.txt")[0]
    for i in range(len(datastream)):
        current_signal = datastream[i:i + 14]
        if len(set(current_signal)) == 14:
            print(i + 14)
            break


def day7():
    class File:
        def __init__(self, name, size, parent):
            self.name = name
            self.size = size
            self.sub_files = []
            self.parent = parent

        def get_size(self):
            if self.size > 0:
                return self.size
            else:
                size = 0
                for subfile in self.sub_files:
                    if type(subfile) == File:
                        print(subfile.name)
                        size += subfile.get_size()
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
                        current_directory = File(parts[2], 0, current_directory)
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
        if dir_delete_min <= directory.get_size() < current_min:
            current_min = directory.get_size()
    print(current_min)


def day8():
    def check_vis(tree_value, list_of_other_trees):
        for other_tree in list_of_other_trees:
            if other_tree >= tree_value:
                return False
        return True

    lines = read_input("day8.txt")[:-1]
    visible_grid = [[False for _ in range(len(lines))] for _ in range(len(lines[0]))]
    for y in range(len(lines)):
        for x in range(len(lines[0])):
            tree_val = int(lines[y][x])
            L = list(map(int, list(lines[y][:x])))
            R = list(map(int, list(lines[y][x + 1:])))
            U = list(map(lambda i: int(i[x]), lines[:y]))
            D = list(map(lambda i: int(i[x]), lines[y + 1:]))
            for direction in L, R, U, D:
                direction.append(-1)

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

    def calc_score(x, y):
        tree_height = int(lines[y][x])
        score = 1
        L = list(map(int, list(lines[y][:x])))[::-1]
        R = list(map(int, list(lines[y][x + 1:])))
        U = list(map(lambda i: int(i[x]), lines[:y]))[::-1]
        D = list(map(lambda i: int(i[x]), lines[y + 1:]))
        for direction in L, R, U, D:
            dir_score = 0
            for dir_tree in direction:
                dir_score += 1
                if dir_tree >= tree_height:
                    break
            score *= dir_score
        return score

    best_score = 0
    for y in range(len(lines)):
        for x in range(len(lines[0])):
            tree_score = calc_score(x, y)
            best_score = tree_score if tree_score > best_score else best_score
    print(best_score)


def day9():
    h = (0, 0)
    t = (0, 0)
    visited = set()
    moves = read_input("day9.txt")
    dirs = {"R": (1, 0), "U": (0, 1), "L": (-1, 0), "D": (0, -1)}
    for move in moves:
        if move == "":
            break
        direction, steps = move.split(" ")
        for step in range(int(steps)):
            h = (h[0] + dirs[direction][0], h[1] + dirs[direction][1])
            rope_vector = (h[0] - t[0], h[1] - t[1])
            if abs(rope_vector[0]) == 2 or abs(rope_vector[1]) == 2:
                tail_move = tuple(
                    map(lambda j: 0 if rope_vector[j] == 0 else rope_vector[j] / abs(rope_vector[j]), [0, 1]))
                t = (t[0] + tail_move[0], t[1] + tail_move[1])
                visited.add(t)
    print(len(visited))
    visited = set()

    class Knot:
        def __init__(self):
            self.position = (0, 0)

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
                cur_knot = knots[i + 1]
                last_knot = knots[i]
                rope_vector = (last_knot.position[0] - cur_knot.position[0],
                               last_knot.position[1] - cur_knot.position[1])
                if abs(rope_vector[0]) == 2 or abs(rope_vector[1]) == 2:
                    tail_move = tuple(
                        map(lambda j: 0 if rope_vector[j] == 0 else rope_vector[j] / abs(rope_vector[j]), [0, 1]))
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
        cycle += 1
        current_draw_pos = (cycle - 1) % 40
        if X - 1 <= current_draw_pos <= X + 1 and cycle <= 240:
            image += "#"
        elif cycle <= 240:
            image += "."
        if cycle % 40 == 20:
            signal_strength = X * cycle
            signal_strengths.append(signal_strength)
        if cycle % 40 == 0:
            image += "\n"
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
    f = open("inputs/day11.txt", "r")
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
                looking_at = looking_at * int(self.operations[1]) if self.operations[
                                                                         1] != "old" else looking_at * looking_at
            if not part_2:
                looking_at = looking_at // 3
            else:
                looking_at = looking_at % part_2_divisor
            return looking_at

        def check_item(self, item):
            if item % self.test_divisor == 0:
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
            if heightmap[y][x] == "S":
                start = (x, y)
            elif heightmap[y][x] == "E":
                end = (x, y)
            path_map[(x, y)] = []
            for direction in [[0, 1], [1, 0], [-1, 0], [0, -1]]:
                if 0 <= x + direction[0] < len(heightmap[0]) and 0 <= y + direction[1] < len(heightmap):
                    if ord(get_height(x, y)) + 1 >= ord(get_height(x + direction[0], y + direction[1])):
                        path_map[(x, y)].append((x + direction[0], y + direction[1]))
    distances = {start: 0}
    queue = [start]
    seen = set()
    counter = 0
    while len(queue) > 0:
        current = queue.pop(0)
        seen.add(current)
        for adj_cell in path_map[current]:
            if adj_cell in distances.keys():
                distances[adj_cell] = min([distances[adj_cell], distances[current] + 1])
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
                end = (x, y)
            path_map[(x, y)] = []
            for direction in [[0, 1], [1, 0], [-1, 0], [0, -1]]:
                if 0 <= x + direction[0] < len(heightmap[0]) and 0 <= y + direction[1] < len(heightmap):
                    if ord(get_height(x, y)) <= ord(get_height(x + direction[0], y + direction[1])) + 1:
                        path_map[(x, y)].append((x + direction[0], y + direction[1]))
    distances = {end: 0}
    queue = [end]
    dist = 999999
    seen = set()
    while len(queue) > 0:
        current = queue.pop(0)
        seen.add(current)
        for adj_cell in path_map[current]:
            current_height = get_height(current[0], current[1])
            if adj_cell in distances.keys():
                if distances[adj_cell] > distances[current] + 1:
                    distances[adj_cell] = distances[current] + 1
                    queue.append(adj_cell)
            else:
                distances[adj_cell] = distances[current] + 1
                queue.append(adj_cell)
        if get_height(current[0], current[1]) == "a" and distances[current] < dist:
            dist = distances[current]
            print(dist)
    print(f"{len(seen)}/{len(heightmap) * len(heightmap[0])}")


def day13():
    data = read_input("day13.txt")
    pairs = []
    pair = []
    for packet in data:
        if len(pair) < 2 and packet != "":
            pair.append(eval(packet))
        elif len(pair) == 2:
            pairs.append(pair)
            pair = []
    pairs.append(pair)
    index_sum = 0
    index = 0

    def test(a, b):
        if type(a) == int and type(b) == int:
            if a < b:
                return "PASS"
            elif a == b:
                return "CONTINUE"
            else:
                return "FAIL"
        elif type(a) == list and type(b) == list:
            comparing = True
            i = 0
            while comparing:
                if i == len(a) and i == len(b):
                    return "CONTINUE"
                elif i == len(a) and i < len(b):
                    return "PASS"
                elif i < len(a) and i == len(b):
                    return "FAIL"
                else:
                    subresult = test(a[i], b[i])
                    if subresult == "CONTINUE":
                        i += 1
                    else:
                        return subresult

        else:
            a = [a] if type(a) == int else a
            b = [b] if type(b) == int else b
            return test(a, b)

    organized_packets = []
    for pair in pairs:
        index += 1
        if test(pair[0], pair[1]) == "PASS":
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
        for i in range(len(organized_packets) - 1):
            if test(organized_packets[i], organized_packets[i + 1]) == "FAIL":
                test_len = len(organized_packets)
                a = organized_packets[i]
                b = organized_packets[i + 1]
                organized_packets[i + 1] = a.copy()
                organized_packets[i] = b.copy()
                if test_len != len(organized_packets):
                    print("packet recomposition broken")
                still_solving = True
    decoder_key = 1
    for i in range(len(organized_packets) - 1):
        if test(organized_packets[i], organized_packets[i + 1]) == "FAIL":
            print("Didnt sort")
        elif organized_packets[i] == [[2]] or organized_packets[i] == [[6]]:
            decoder_key *= i + 1
    print(decoder_key)

    print(f"{index_sum}/{sum(list(range(1, len(pairs) + 1)))}")

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

    sand_positions = set()
    Sand_pouring = True

    def can_move(pos):
        if pos[1] == min_y + 1 and part2:
            return False
        elif (pos[0], pos[1] + 1) not in rock_positions and (pos[0], pos[1] + 1) not in sand_positions:
            return (pos[0], pos[1] + 1)
        elif (pos[0] - 1, pos[1] + 1) not in rock_positions and (pos[0] - 1, pos[1] + 1) not in sand_positions:
            return (pos[0] - 1, pos[1] + 1)
        elif (pos[0] + 1, pos[1] + 1) not in rock_positions and (pos[0] + 1, pos[1] + 1) not in sand_positions:
            return (pos[0] + 1, pos[1] + 1)
        else:
            return False

    # draw sand castle

    def view(cur_sand=None):
        if not cur_sand:
            y1 = max_y - 15
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

    while Sand_pouring:
        sand_position = (500, 0)
        in_abyss = False
        while can_move(sand_position):
            sand_position = can_move(sand_position)
            # view(sand_position)
            if sand_position[1] > min_y and not part2:
                in_abyss = True
                break
        if not in_abyss:
            sand_positions.add(sand_position)
        elif in_abyss:
            break
        elif (500, 0) in sand_positions:
            break

        if len(sand_positions) % 100 == 0:
            # view()
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
    sand_positions = {(500, 0)}
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

    def dist(a, b):
        if type(a) == Zone and type(b) == Zone:
            return abs(a.center[0] - b.center[0]) + abs(a.center[1] - b.center[1])
        elif type(a) == tuple and type(b) == tuple:
            return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def intersect(a, b):
        if dist(a, b) < a.radius + b.radius:
            return True
        else:
            return False

    def adjacent(a, b):
        if dist(a, b) == a.radius + b.radius:
            return True
        else:
            return False

    for sensor in sensors:
        parts = sensor.split(" ")
        sensor_pos = (int(parts[2].strip("x=,")), int(parts[3].strip("y=:")))
        beacon_pos = (int(parts[8].strip("x=,")), int(parts[9].strip("y=")))
        man_dist = abs(sensor_pos[0] - beacon_pos[0]) + abs(sensor_pos[1] - beacon_pos[1])
        if abs(sensor_pos[1] - target_row) <= man_dist:
            excess = man_dist - abs(sensor_pos[1] - target_row)
            new_range = (sensor_pos[0] - excess, sensor_pos[0] + excess)
            ranges.append(new_range)
        if beacon_pos[1] == target_row:
            on_target_row.add(beacon_pos[0])
    ranges.sort()
    for i in range(len(ranges)):
        for j in range(i + 1, len(ranges)):
            if ranges[i][1] >= ranges[j][0]:
                ranges[j] = (ranges[i][1] + 1, ranges[j][1])
        if ranges[i][0] <= ranges[i][1]:
            cant_be += ranges[i][1] - ranges[i][0] + 1
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

    # for quadruple in itertools.combinations(circles, 4):
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
        data.append({"sensor": sensor_pos, "man_dist": man_dist})

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
            if next_range[0] < x < next_range[1]:
                x = next_range[1] + 1
            elif next_range[0] == x + 1:
                print(f"({x},{y})")
                print(x * 4000000 + y)
        if y % 100000 == 0:
            print(f"{100 * y / 4000000}")

    # day 15 part 2 try 1 16000002765172 too high
    #               try 2 10884459367718 !!!


def day16():
    tunnel_info = read_input("day16.txt")
    valves = dict()

    for line in tunnel_info:
        new_valve = dict()
        new_valve["name"] = line.split(" ")[1]
        new_valve["rate"] = int(line.split(" ")[4].strip("rate=;"))
        new_valve["connections"] = list(map(lambda i: i.strip(","), line.split(" ")[9:]))
        valves[new_valve["name"]] = new_valve
    important_valves = []
    for valve in valves.keys():
        if valves[valve]["rate"] > 0:
            important_valves.append(valve)
    print(f"valves we are going to look at: {important_valves}")

    # initialize distances map with all the neighboring zones or "infinite" dist
    distances = dict()
    for valve in valves.keys():
        distances[valve] = {valve: 0}
        for other_valve in valves.keys():
            if other_valve in valves[valve]["connections"]:
                distances[valve][other_valve] = 1
            else:
                distances[valve][other_valve] = 9999999

    def find_distances(start_node):
        queue = list(valves[start_node]["connections"])
        visited = set()
        while len(queue) > 0:
            queue = sorted(queue, key=lambda i: distances[start_node][i])
            v = queue.pop(0)
            for v_neighbor in valves[v]["connections"]:
                if distances[start_node][v_neighbor] > distances[start_node][v] + 1:
                    distances[start_node][v_neighbor] = distances[start_node][v] + 1
                if v_neighbor not in visited and v_neighbor not in queue:
                    queue.append(v_neighbor)
            visited.add(v)

    find_distances("AA")
    for valve in important_valves:
        find_distances(valve)

    # simplified the problem by determining the distances between all the important positions, we can ignore all the
    # inbetween steps and just focus on how long it takes to move between different valve positions

    decision_paths = [["AA"]]

    def time_cost(path):
        t = 0
        for i in range(1, len(path)):
            t += 1 + distances[path[i - 1]][path[i]]
        return t

    def get_score(path, part1=True):
        score = 0
        time_remaining = 30 if part1 else 26
        for i in range(1, len(path)):
            time_remaining -= distances[path[i - 1]][path[i]] + 1
            score += valves[path[i]]["rate"] * time_remaining
        return score

    best_score = 0
    while len(decision_paths) > 0:
        decision_paths = sorted(decision_paths, key=lambda i: get_score(i), reverse=True)
        cur_path = decision_paths.pop(0)
        if get_score(cur_path) > best_score:
            best_score = get_score(cur_path)
            print(best_score)
        new_paths = []
        for valve in important_valves:
            if valve not in cur_path:
                new_path = cur_path + [valve]
                new_paths.append(new_path)
        for path in new_paths:
            if time_cost(path) <= 30:
                decision_paths.append(path)

    print(f"part1: {best_score}")

    # reset for part2
    best_score = 0
    decision_paths = [[["AA"], ["AA"]]]

    def get_score_part2(paths):
        return sum(map(lambda i: get_score(i, part1=False), paths))

    j = 0
    while len(decision_paths) > 0:
        j += 1
        if j % 10000 == 0:
            print(f"number of decisions to think about: {len(decision_paths)}")
            decision_paths = sorted(decision_paths, key=lambda i: get_score_part2(i), reverse=True)
        cur_paths = decision_paths.pop(0)
        if get_score_part2(cur_paths) > best_score:
            best_score = get_score_part2(cur_paths)
            print(f"best_score {best_score}")
        new_paths = []
        for valve in important_valves:
            if valve not in cur_paths[0] and valve not in cur_paths[1]:
                new_path_me = cur_paths[0] + [valve]
                new_path_elephant = cur_paths[1] + [valve]
                new_paths.append([new_path_me, cur_paths[1]])
                new_paths.append([cur_paths[0], new_path_elephant])
        for path in new_paths:
            if max(time_cost(path[0]), time_cost(path[1])) <= 30:
                decision_paths.append(path)
    # 2797 too low 3338 too high
    # didnt really ever optimize this enough, just ran it until a good score came out lol
    # probably could note the best score for certain positions in certain times and ignore worse situations


def day17():
    jets = read_input("day17.txt")[0]

    shape_types = ["-",
                   "+",
                   "L",
                   "|",
                   "o"]

    def calc_new_max_height():
        for tile in floor_blocks:
            if tile[1]+1>day17.max_height:
                day17.max_height = tile[1]+1

    class Block:
        def __init__(self, shape):
            self.falling = True
            self.shape = {"-": [(2, 3), (3, 3), (4, 3), (5, 3)],
                            "+": [(2, 4), (3, 4), (4, 4), (3, 5), (3, 3)],
                            "L": [(2, 3), (3, 3), (4, 3), (4, 4), (4, 5)],
                            "|": [(2, 3), (2, 4), (2, 5), (2, 6)],
                            "o": [(2, 3), (3, 3), (2, 4), (3, 4)]}[shape]
            for i in range(len(self.shape)):
                self.shape[i] = (self.shape[i][0], self.shape[i][1]+day17.max_height)

        def blow(self, direction):
            vec = -1 if direction == "<" else 1
            new_shape = []
            for tile in self.shape:
                new_block = (tile[0]+vec, tile[1])
                if not 0 <= new_block[0] <= 6:
                    return "collide"
                elif new_block in floor_blocks:
                    return "collide"
                new_shape.append(new_block)
            self.shape = new_shape.copy()

        def fall(self):
            new_shape = []
            for block in self.shape:
                new_block = (block[0], block[1]-1)
                if new_block in floor_blocks:
                    self.falling = False
                    self.freeze()
                    return "collide"
                new_shape.append(new_block)
            self.shape = new_shape.copy()

        def freeze(self):
            for block in self.shape:
                if block[1]+1>day17.max_height:
                    day17.max_height = block[1]+1
                floor_blocks.add(block)
            #calc_new_max_height()

    def view(cur_block):
        for y in range(day17.max_height + 2, day17.max_height - 10, -1):
            cur_line = "|"
            for x in range(7):
                if (x, y) in floor_blocks:
                    cur_line+="#"
                elif (x, y) in cur_block.shape:
                    cur_line+="@"
                else:
                    cur_line+="."
            cur_line += "|"
            print(cur_line)


    # initialize state of system
    floor_blocks = {(i, -1) for i in range(7)}
    day17.max_height = 0
    turn = 0

    for rock in range(2022):

        new_rock = Block(shape_types[rock%5])

        #view(new_rock)
        #print("\n")

        while new_rock.falling:
            new_rock.blow(jets[turn%len(jets)])
            new_rock.fall()
            turn += 1
            #view(new_rock)
            #print("\n")

    print(day17.max_height)
    #part 1 try 1 3084 too low (blowing wrong way)
    # for part 2, i need to figure out at what point the cycle keeps repeating and then just multiply

    # initialize state of system
    floor_blocks = {(i, -1) for i in range(7)}
    day17.max_height = last_max_height = 0
    turn = last_turn_of_last_rock_cycle = 0
    rock_count = last_rock_count = 0
    cycle_0_rock_count = None
    cycle_0_height = None
    cycle_1_rock_count = None
    cycle_1_height = None
    goal = 1000000000000
    while rock_count<goal:
        if rock_count % 1000 == 0:
            print(rock_count)
        for rock in shape_types:
            new_rock = Block(rock)
            rock_count += 1
            while new_rock.falling:
                new_rock.blow(jets[turn%len(jets)])
                new_rock.fall()
                turn += 1

        if last_turn_of_last_rock_cycle % len(jets) > turn % len(jets):
            print(f"rocks {rock_count}, rocks delta: {rock_count-last_rock_count} h: {day17.max_height}, h_delta: {day17.max_height - last_max_height}")
            if not cycle_0_rock_count:
                cycle_0_rock_count = rock_count
                cycle_0_height = day17.max_height
            elif not cycle_1_rock_count and cycle_0_rock_count:
                print("starting jump")
                # now do a big mega push that jumps ahead a bunch of cycles.
                cycle_1_rock_count = rock_count
                cycle_1_height = day17.max_height
                dist_remaining = goal - cycle_1_rock_count
                number_of_cycles_in_dist_remaining = dist_remaining//(cycle_1_rock_count-cycle_0_rock_count)
                rock_count += number_of_cycles_in_dist_remaining*(cycle_1_rock_count-cycle_0_rock_count)
                day17.max_height += number_of_cycles_in_dist_remaining*(cycle_1_height-cycle_0_height)
                # slide all of our blocks on up
                size_of_jump = day17.max_height - cycle_1_height
                new_blocks = set()
                for block in floor_blocks:
                    new_blocks.add((block[0],block[1]+size_of_jump))
                floor_blocks = new_blocks
                print("jump finished, running last leg")
        last_turn_of_last_rock_cycle = turn
    print(day17.max_height)


def day18_from_scratch():
    inp = read_input("day18.txt")
    lava_cubes = set()
    air = set()
    for line in inp:
        lava_cubes.add(tuple(map(int, line.split(","))))
    dirs = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [-1, 0, 0], [0, -1, 0], [0, 0, -1]]
    surface_area = 0
    for cube in lava_cubes:
        for v in dirs:
            target = (cube[0] + v[0], cube[1] + v[1], cube[2] + v[2])
            if target not in lava_cubes:
                surface_area += 1
                air.add(target)

    # expand air
    to_add = set()
    for cube in air:
        for v in dirs:
            target = (cube[0] + v[0], cube[1] + v[1], cube[2] + v[2])
            if target not in lava_cubes:
                to_add.add(target)
    for c in to_add:
        air.add(c)

    print(f"part 1 total surface area: {surface_area}")
    print(f"number of cubes of adjacent air: {len(air)}")
    zones = []
    while len(air) > 0:
        new_zone = {air.pop()}
        queue = new_zone.copy()
        while len(queue) > 0:
            cube = queue.pop()
            for v in dirs:
                target = (cube[0] + v[0], cube[1] + v[1], cube[2] + v[2])
                if target in air:
                    new_zone.add(target)
                    queue.add(target)
                    air.remove(target)
        zones.append(new_zone)
    print(f"number of bubbles+outside surface = {len(zones)}")
    air_check = 0
    for zone in zones:
        air_check += len(zone)
    print(f"number of cubes of air in zones check: {air_check}")
    max_zone = None
    pt = list(lava_cubes)[0]
    x_max = -10
    for zone in zones:
        for cube in zone:
            if cube[0] > x_max:
                x_max = cube[0]
                max_zone = zone

    outside_surface = 0
    print(f"cubes of air on surface: {len(max_zone)}")
    for cube in max_zone:
        for v in dirs:
            target = (cube[0] + v[0], cube[1] + v[1], cube[2] + v[2])
            if target in lava_cubes:
                outside_surface += 1
    print(outside_surface)
    # 461 too low


def day19_try1():
    blueprint_texts = read_input("day19test.txt")
    blueprints = []
    for blueprint_line in blueprint_texts:
        parts = blueprint_line.split(" ")
        new_blueprint = {"num": int(parts[1].strip(":")), "ore bot": {"ore": int(parts[6])},
                         "clay bot": {"ore": int(parts[12])},
                         "obsidian bot": {"ore": int(parts[18]), "clay": int(parts[21])},
                         "geode bot": {"ore": int(parts[27]), "obsidian": int(parts[30])}}
        blueprints.append(new_blueprint)

    qualities = []
    tuple_keys = {"ore bot": 0, "clay bot": 1, "obsidian bot": 2, "geode bot": 3,
                  "ore": 4, "clay": 5, "obsidian": 6, "geodes": 7}
    for blueprint in blueprints:
        best_quality = 0
        # initial conditions4
        # going to try and represent states as tuples instead of class objects
        max_necessary_ore_bot = max(blueprint["ore bot"]["ore"],
                                    blueprint["clay bot"]["ore"],
                                    blueprint["obsidian bot"]["ore"],
                                    blueprint["geode bot"]["ore"])
        max_necessary_clay_bot = blueprint["obsidian bot"]["clay"]
        max_necessary_obsidian_bot = blueprint["geode bot"]["obsidian"]
        initial = (1, 0, 0, 0, 0, 0, 0, 0, None)

        # (ore bot, clay bot, obs bot, geo bot, ore, clay, obs, geo, next_build)

        def build(state_tuple):
            # goes through each possible type of bot and creates a new tuple representing that state of the world
            # then we make a set of those states so we can mash them
            # into our bigger pile of states and ignore intersecting
            # possibilities of states
            states = set()
            if blueprint["geode bot"]['ore'] <= state_tuple[4] and blueprint['geode bot']['obsidian'] <= state_tuple[6]:
                new_tuple = list(state_tuple)
                new_tuple[4] -= blueprint['geode bot']['ore']
                new_tuple[6] -= blueprint['geode bot']['obsidian']
                new_tuple[8] = "geode bot"
                states.add(tuple(new_tuple))
            if blueprint["obsidian bot"]['ore'] <= state_tuple[4] and blueprint['obsidian bot']['clay'] <= state_tuple[
                5] and state_tuple[2] < max_necessary_obsidian_bot:
                new_tuple = list(state_tuple)
                new_tuple[4] -= blueprint['obsidian bot']['ore']
                new_tuple[5] -= blueprint['obsidian bot']['clay']
                new_tuple[8] = "obsidian bot"
                states.add(tuple(new_tuple))
            if blueprint["clay bot"]['ore'] <= state_tuple[4] and state_tuple[1] < max_necessary_clay_bot:
                new_tuple = list(state_tuple)
                new_tuple[4] -= blueprint['clay bot']['ore']
                new_tuple[8] = "clay bot"
                states.add(tuple(new_tuple))
            if blueprint["ore bot"]['ore'] <= state_tuple[4] and state_tuple[0] < max_necessary_ore_bot:
                new_tuple = list(state_tuple)
                new_tuple[4] -= blueprint['ore bot']['ore']
                new_tuple[8] = "ore bot"
                states.add(tuple(new_tuple))

            def can_make():
                can_make_everything = True
                if state_tuple[4] < max(blueprint["ore bot"]["ore"],
                                        blueprint["clay bot"]["ore"],
                                        blueprint["obsidian bot"]["ore"],
                                        blueprint["geode bot"]["ore"]):
                    can_make_everything = False
                elif state_tuple[tuple_keys["clay bot"]] > 0 and blueprint["obsidian bot"]["clay"] > state_tuple[5]:
                    can_make_everything = False
                elif state_tuple[6] < blueprint["geode bot"]["obsidian"] and state_tuple[2] > 0:
                    can_make_everything = False
                return can_make_everything

            if not can_make():
                states.add(state_tuple)
            return states

        def mine(state_tuple):
            lis = list(state_tuple)
            lis[tuple_keys["ore"]] += lis[tuple_keys["ore bot"]]
            lis[tuple_keys["clay"]] += lis[tuple_keys["clay bot"]]
            lis[tuple_keys["obsidian"]] += lis[tuple_keys["obsidian bot"]]
            lis[tuple_keys["geodes"]] += lis[tuple_keys["geode bot"]]
            return tuple(lis)

        def finish_building(state_tuple):
            lis = list(state_tuple)
            if lis[8]:
                lis[tuple_keys[lis[8]]] += 1
            lis[8] = None
            return tuple(lis)

        possible_states = set([initial])

        for each_minute in range(24):
            print(f"{each_minute}: {len(possible_states)}")
            states_building_selected = set()
            for state in possible_states:
                states_building_selected = states_building_selected.union(build(state))
            print(f"selected possible build configurations: {len(states_building_selected)}")
            states_mined = set()
            geode_leader = (0, 0, 0, 0, 0, 0, 0, 0)
            for state in states_building_selected:
                cur_state_mined = mine(state)
                if cur_state_mined not in states_mined:
                    if cur_state_mined[tuple_keys["geodes"]] >= geode_leader[tuple_keys["geodes"]]:
                        if cur_state_mined[tuple_keys["geode bot"]] > geode_leader[tuple_keys["geode bot"]]:
                            geode_leader = cur_state_mined
                        states_mined.add(cur_state_mined)
                    else:
                        geode_leader_minimum = geode_leader[7] + geode_leader[3] * (23 - each_minute)
                        cur_state_maximum = cur_state_mined[7] + int(
                            (23 - each_minute) * (2 * cur_state_mined[3] + (23 - each_minute) + 1) / 2)
                        if cur_state_maximum >= geode_leader_minimum:
                            states_mined.add(cur_state_mined)
            print(f"mined and pruned states {len(states_mined)}")
            next_states = set()
            for state in states_mined:
                next_states.add(finish_building(state))
            print(f"finished buildings in every state {len(next_states)}")
            possible_states = next_states.copy()
            print(f"geodes: {geode_leader[7]}")
        for state in possible_states:
            if blueprint['num'] * state[tuple_keys["geodes"]] >= best_quality:
                best_quality = blueprint['num'] * state[tuple_keys["geodes"]]
                print(best_quality)
        qualities.append(best_quality)
        print(best_quality / blueprint["num"])
    print(sum(qualities))


def day19():
    blueprint_texts = read_input("day19.txt")
    blueprints = []
    for blueprint_line in blueprint_texts:
        parts = blueprint_line.split(" ")
        new_blueprint = {"num": int(parts[1].strip(":")), "ore bot": {"ore": int(parts[6])},
                         "clay bot": {"ore": int(parts[12])},
                         "obsidian bot": {"ore": int(parts[18]), "clay": int(parts[21])},
                         "geode bot": {"ore": int(parts[27]), "obsidian": int(parts[30])}}
        blueprints.append(new_blueprint)

    # class for holding information about state of bot simulations
    class State:
        def __init__(self, blueprint, goal, time_remaining=24, bots=(1, 0, 0, 0), material=(0, 0, 0, 0)):
            self.ore_bots, self.clay_bots, self.obsidian_bots, self.geode_bots = bots
            self.ore, self.clay, self.obsidian, self.geode = material
            self.goal = goal
            self.blueprint = blueprint
            self.time_remaining = time_remaining

            # first determine the most bots we could possibly need of each type
            self.max_necessary_ore_bot = max(blueprint["ore bot"]["ore"],
                                             blueprint["clay bot"]["ore"],
                                             blueprint["obsidian bot"]["ore"],
                                             blueprint["geode bot"]["ore"])
            self.max_necessary_clay_bot = blueprint["obsidian bot"]["clay"]
            self.max_necessary_obsidian_bot = blueprint["geode bot"]["obsidian"]

        def get_bots(self):
            return self.ore_bots, self.clay_bots, self.obsidian_bots, self.geode_bots

        def get_material(self):
            return self.ore, self.clay, self.obsidian, self.geode

        def potential_ore(self):
            return self.time_remaining * self.ore_bots + self.ore

        def potential_clay(self):
            return self.time_remaining * self.clay_bots + self.clay

        def potential_obsidian(self):
            return self.time_remaining * self.obsidian_bots + self.obsidian

        def potential_geode(self):
            return self.time_remaining * self.geode_bots + self.geode

        def max_potential_geode(self):
            geodes = self.geode
            g_bots = self.geode_bots
            for _ in range(self.time_remaining):
                geodes += g_bots
                g_bots += 1
            return geodes

        def can_get_ore_bot(self):
            return self.potential_ore() >= self.blueprint["ore bot"]["ore"]

        def can_get_clay_bot(self):
            return self.potential_ore() >= self.blueprint["clay bot"]["ore"]

        def can_get_obsidian_bot(self):
            return self.potential_ore() >= self.blueprint["obsidian bot"]["ore"] and \
                   self.potential_clay() >= self.blueprint["obsidian bot"]["clay"]

        def can_get_geode_bot(self):
            return self.potential_ore() >= self.blueprint["geode bot"]["ore"] and \
                   self.potential_obsidian() >= self.blueprint["geode bot"]["obsidian"]

        def get_next_bots(self):
            bot_list = []
            if self.ore_bots < self.max_necessary_ore_bot and self.can_get_ore_bot():
                bot_list.append("ore bot")
            if self.clay_bots < self.max_necessary_clay_bot and self.can_get_clay_bot():
                bot_list.append("clay bot")
            if self.clay_bots > 0 and self.obsidian_bots < self.max_necessary_obsidian_bot and self.can_get_obsidian_bot():
                bot_list.append("obsidian bot")
            if self.obsidian_bots > 0 and self.can_get_geode_bot():
                bot_list.append("geode bot")
            return bot_list

        def turns_til_can_afford(self, bot_name):
            min_turns = 0
            for cost_name in self.blueprint[bot_name].keys():
                if cost_name == 'ore':
                    needed_ore = self.blueprint[bot_name][cost_name] - self.ore
                    if needed_ore > 0:
                        turns = np.ceil(needed_ore / self.ore_bots)
                        min_turns = turns if turns > min_turns else min_turns
                if cost_name == 'clay':
                    needed_clay = self.blueprint[bot_name][cost_name] - self.clay
                    if needed_clay > 0:
                        turns = np.ceil(needed_clay / self.clay_bots)
                        min_turns = turns if turns > min_turns else min_turns
                if cost_name == 'obsidian':
                    needed_obsidian = self.blueprint[bot_name][cost_name] - self.obsidian
                    if needed_obsidian > 0:
                        turns = np.ceil(needed_obsidian / self.obsidian_bots)
                        min_turns = turns if turns > min_turns else min_turns
            return int(min_turns)

        def build(self):
            if self.goal == "clay bot":
                self.clay_bots += 1
            elif self.goal == "ore bot":
                self.ore_bots += 1
            elif self.goal == "geode bot":
                self.geode_bots += 1
            elif self.goal == "obsidian bot":
                self.obsidian_bots += 1
            for cost_name in self.blueprint[self.goal].keys():
                if cost_name == 'ore':
                    self.ore -= self.blueprint[self.goal][cost_name]
                elif cost_name == 'clay':
                    self.clay -= self.blueprint[self.goal][cost_name]
                elif cost_name == 'obsidian':
                    self.obsidian -= self.blueprint[self.goal][cost_name]
            self.goal = None

        def accomplish_goal(self):
            turns = self.turns_til_can_afford(self.goal)
            for i in range(turns + 1):
                self.ore += self.ore_bots
                self.clay += self.clay_bots
                self.obsidian += self.obsidian_bots
                self.geode += self.geode_bots
                self.time_remaining -= 1
            self.build()

    def find_best_blueprint_strategy(blueprint, time=24, output='quality'):
        quality = 0
        queue = [State(blueprint, goal=None, time_remaining=time)]
        i = 1
        while len(queue) > 0:
            i += 1
            if i % 100 == 0:
                if i % 1000 == 0:
                    print(f"#{blueprint['num']}: {len(queue)} states")
                queue = sorted(queue, key=lambda j: j.geode, reverse=True)
            cur_state = queue.pop(0)
            if cur_state.potential_geode() > quality:
                quality = cur_state.potential_geode()
            if cur_state.max_potential_geode() > quality:
                for possible_next_bot in cur_state.get_next_bots():
                    new_state = State(blueprint, goal=possible_next_bot, time_remaining=cur_state.time_remaining,
                                      bots=cur_state.get_bots(), material=cur_state.get_material())
                    new_state.accomplish_goal()
                    queue.append(new_state)
        print(f"quality of BP#{blueprint['num']}: {quality * blueprint['num']}")
        if output == 'quality':
            return quality * blueprint["num"]
        elif output == 'geode':
            return quality

    total_quality = 0
    for bp in blueprints:
        total_quality += find_best_blueprint_strategy(bp)
    print(f"part1 {total_quality}")

    quality_multiple = 1
    for bp in blueprints[0:3]:
        quality_multiple *= find_best_blueprint_strategy(bp, time=32, output='geode')
    print(f"part2 {quality_multiple}")


def day20():
    nums = read_input("day20.txt")
    nums = list(map(int, nums))
    part2 = True
    if part2:
        nums = list(map(lambda i: i * 811589153, nums))
    indices = list(range(len(nums)))
    for i in list(range(len(nums))) * 10:
        # print(nums)
        i_x = indices.index(i)
        num = nums[i_x]
        for _ in range(abs(num) % (len(nums) - 1)):
            if num > 0:
                swap_with_num = nums[(i_x + 1) % len(nums)]
                swap_with_index = indices[(i_x + 1) % len(nums)]
                nums[i_x] = swap_with_num
                nums[(i_x + 1) % len(nums)] = num
                indices[i_x] = swap_with_index
                indices[(i_x + 1) % len(nums)] = i
                i_x = (i_x + 1) % len(nums)
            if num < 0:
                swap_with_num = nums[(i_x - 1) % len(nums)]
                swap_with_index = indices[(i_x - 1) % len(nums)]
                nums[i_x] = swap_with_num
                nums[(i_x - 1) % len(nums)] = num
                indices[i_x] = swap_with_index
                indices[(i_x - 1) % len(nums)] = i
                i_x = (i_x - 1) % len(nums)

    zero_loc = nums.index(0)
    # print(nums)
    print(
        nums[(zero_loc + 1000) % len(nums)] + nums[(zero_loc + 2000) % len(nums)] + nums[(zero_loc + 3000) % len(nums)])
    # day20part1 try 1: -7210 (incorrect)
    # part1: correct: 7225


def day21():
    monkey_list = read_input("day21.txt")
    monkeys = dict()
    part2 = True

    class Monkey:
        def __init__(self, monkey_line):
            parts = monkey_line.split(" ")
            self.name = parts[0].strip(":")
            self.value = None
            self.a = None
            self.b = None
            self.op = None
            if len(parts[1:]) == 1:
                self.value = int(parts[1])
            else:
                self.a = parts[1]
                self.b = parts[3]
                self.op = parts[2]
                if part2 and self.name == "root":
                    self.op = "=="
            if part2 and self.name == "humn":
                self.value = "UNKNOWN"

        def get_value(self, other_monkeys):
            if not self.value:
                self.value = "(" + str(other_monkeys[self.a].get_value(other_monkeys)) + self.op + str(
                    other_monkeys[self.b].get_value(other_monkeys)) + ")"
                if not part2 or "UNKNOWN" not in self.value:
                    self.value = eval(self.value)
            return self.value

    for monkey in monkey_list:
        new_monkey = Monkey(monkey)
        monkeys[new_monkey.name] = new_monkey
    # part1
    # print(monkeys["root"].get_value(monkeys))
    left = monkeys[monkeys["root"].a].get_value(monkeys)
    right = monkeys[monkeys["root"].b].get_value(monkeys)
    while left != "UNKNOWN":
        print(f"{left}={right}")
        if left[0] == "(" and left[-1] == ")":
            left = left[1:-1]
        elif left[0] == "(":
            for i in range(1, len(left)):
                if left[-i] in "*/-+":
                    op = left[-i]
                    num = left[(-i) + 1:]
                    break
            op = {"/": "*",
                  "*": "/",
                  "+": "-",
                  "-": "+"}[op]
            right = eval(str(right) + op + num)
            left = left[:-i]
        elif left[-1] == ")":
            for i in range(len(left)):
                if left[i] in "*/-+":
                    op = left[i]
                    num = left[:i]
                    break

            if op in "*+":
                op = {"*": "/",
                      "+": "-", }[op]
                right = eval(str(right) + op + num)
                left = left[i + 1:]
            elif op == "/":
                right = eval(num + "/" + str(right))
                left = left[i + 1:]
            elif op == "-":
                right = eval(num + "-" + str(right))
                left = left[i + 1:]
        else:
            break
    print(right)
    # part2 try 1 3451534021574, too low
    # part2 try 2 3451534022348 (based on last eval not working? because there isn't a parentheses lol


def day22():
    board = read_input("day22.txt", strip=False)[:-1]
    instructions = list(board[-1])
    board = board[:-1]
    width = len(board[0])
    height = len(board)
    part2 = True
    size = 50
    dirs = {(1, 0): {"R": (0, 1), "L": (0, -1)},
            (0, 1): {"R": (-1, 0), "L": (1, 0)},
            (-1, 0): {"R": (0, -1), "L": (0, 1)},
            (0, -1): {"R": (1, 0), "L": (-1, 0)},
            }

    x = board[0].find(".")
    y = 0
    day22.facing = (1, 0)
    dist = ""

    for row_num in range(len(board)):
        board[row_num] = board[row_num].ljust(width, " ")

    def cube_pos(coords):
        # im probably a moron. but i don't want to try and make the map for this lmao
        if 0 <= coords[0] < width and 0 <= coords[1] < height:
            if board[coords[1]][coords[0]] in ".#":
                return coords[0], coords[1], day22.facing
        # going from 1->6
        if coords[1] == -1 and coords[0] in range(50, 100):
            new_facing = (1, 0)
            return 0, coords[0] - 50 + 150, new_facing
        # going from 1->5
        elif coords[0] == 49 and coords[1] in range(0, 50):
            new_facing = (1, 0)
            return 0, 49 - coords[1] + 100, new_facing
        # going from 2->6
        elif coords[1] == -1 and coords[0] in range(100, 150):
            new_facing = (0, -1)
            return coords[0] - 100, 199, new_facing
        # going from 2->4
        elif coords[0] == 150 and coords[1] in range(0, 50):
            new_facing = (-1, 0)
            return 99, 49 - coords[1] + 100, new_facing
        # going from 2->3
        elif coords[1] == 50 and coords[0] in range(100, 150):
            new_facing = (-1, 0)
            return 99, coords[0] - 100 + 50, new_facing
        # going from 3->2
        elif coords[0] == 100 and coords[1] in range(50, 100):
            new_facing = (0, -1)
            return coords[1] - 50 + 100, 49, new_facing
        # going from 3->5
        elif coords[0] == 49 and coords[1] in range(50, 100):
            new_facing = (0, 1)
            return coords[1] - 50, 100, new_facing
        # going from 4->2
        elif coords[0] == 100 and coords[1] in range(100, 150):
            new_facing = (-1, 0)
            return 149, 49 - (coords[1] - 100), new_facing
        # going from 4->6
        elif coords[1] == 150 and coords[0] in range(50, 100):
            new_facing = (-1, 0)
            return 49, coords[0] - 50 + 150, new_facing
        # going from 5->3
        elif coords[1] == 99 and coords[0] in range(0, 50):
            new_facing = (1, 0)
            return 50, coords[0] + 50, new_facing
        # going from 5->1
        elif coords[0] == -1 and coords[1] in range(100, 150):
            new_facing = (1, 0)
            return 50, 49 - (coords[1] - 100), new_facing
        # going from 6->4
        elif coords[0] == 50 and coords[1] in range(150, 200):
            new_facing = (0, -1)
            return coords[1] - 150 + 50, 149, new_facing
        # going from 6->2
        elif coords[1] == 200 and coords[0] in range(0, 50):
            new_facing = (0, 1)
            return coords[0] + 100, 0, new_facing
        # going from 6->1
        elif coords[0] == -1 and coords[1] in range(150, 200):
            new_facing = (0, 1)
            return coords[1] - 150 + 50, 0, new_facing
        else:
            print(coords)

    def can_move(coords):
        if not part2:
            next_pos = ((coords[0] + day22.facing[0]) % width, (coords[1] + day22.facing[1]) % height)
        else:
            new_x, new_y, new_facing = cube_pos(((coords[0] + day22.facing[0]), (coords[1] + day22.facing[1])))
            next_pos = (new_x, new_y)
        if board[next_pos[1]][next_pos[0]] == ".":
            return next_pos[0], next_pos[1], new_facing
        elif board[next_pos[1]][next_pos[0]] == "#":
            return False
        elif board[next_pos[1]][next_pos[0]] == " ":
            return can_move(next_pos)

    def print_pos():
        print_board = board.copy()
        me = {(-1, 0): "<", (1, 0): ">", (0, 1): "v", (0, -1): "^"}[day22.facing]
        print_board[y] = print_board[y][:x] + me + print_board[y][x + 1:]
        output = ""
        for line in print_board:
            output = output + line + "\n"
        print(output)

    while len(instructions) > 0:
        next_char = instructions.pop(0)
        if next_char not in "LR":
            dist = dist + next_char
        if next_char in "LR" or len(instructions) == 0:
            for _ in range(int(dist)):
                movement = can_move((x, y))
                if movement:
                    x, y, day22.facing = movement
                else:
                    break
            if next_char in "LR":
                day22.facing = dirs[day22.facing][next_char]
            # print(dist+next_char)
            # print_pos()
            dist = ""
    face_val = {(1, 0): 0, (0, 1): 1, (-1, 0): 2, (0, -1): 3}[day22.facing]
    result = (1000 * (y + 1)) + (4 * (x + 1)) + face_val
    print(result)

    # part1 try1 36570 too high
    # fixed last line 36518

    # part2 try1 138022 too low
    # part2 try2 75284, after trying to fix the facing variable
    # part2 try2 143208, fixed the getting turned to match the new face even if we didn't cross to the new face


def day23():
    ground_map = read_input("day23.txt")

    dirs = {"N": (0, -1), "NE": (1, -1), "E": (1, 0), "SE": (1, 1), "S": (0, 1), "SW": (-1, 1), "W": (-1, 0),
            "NW": (-1, -1)}

    choices = {"N": ["N", "NE", "NW"],
               "E": ["NE", "E", "SE"],
               "W": ["W", "SW", "NW"],
               "S": ["S", "SE", "SW"]}
    choice_order = "NSWE"

    part2 = True
    all_elf_positions = set()

    class Elf:
        def __init__(self, x, y):
            self.x = x
            self.y = y
            self.target = None

        def check_surroudings(self, positions):
            neighbors = []
            for k, v in dirs.items():
                if (self.x + v[0], self.y + v[1]) in positions:
                    neighbors.append(k)
            if len(neighbors) > 0:
                for choice in choice_order:
                    choice_good = True
                    for i in choices[choice]:
                        if i in neighbors:
                            choice_good = False
                            break
                    if choice_good:
                        self.target = (self.x + dirs[choice][0], self.y + dirs[choice][1])
                        break
            else:
                self.target = None

        def try_move(self, positions):
            if self.target and positions.count(self.target) == 1:
                self.x, self.y = self.target
            self.target = None

    elves = []

    for y in range(len(ground_map)):
        for x in range(len(ground_map[0])):
            if ground_map[y][x] == "#":
                elves.append(Elf(x, y))
    print(len(elves))

    if not part2:
        for _ in range(10):
            all_elf_positions = set(map(lambda i: (i.x, i.y), elves))
            for elf in elves:
                elf.check_surroudings(all_elf_positions)
            all_targets = list(map(lambda i: i.target, elves))
            for elf in elves:
                elf.try_move(all_targets)
            choice_order = choice_order[1:] + choice_order[0]
        xi, xf, yi, yf = elves[0].x, elves[0].x, elves[0].y, elves[0].y

        for elf in elves:
            if elf.x < xi:
                xi = elf.x
            if elf.x > xf:
                xf = elf.x
            if elf.y < yi:
                yi = elf.y
            if elf.y > yf:
                yf = elf.y
        area = (xf + 1 - xi) * (yf + 1 - yi)
        print(area - len(elves))
        test_output = ""
        for y in range(yi, yf + 1):
            for x in range(xi, xf + 1):
                if (x, y) in set(map(lambda i: (i.x, i.y), elves)):
                    test_output += "#"
                else:
                    test_output += "."
            test_output += "\n"
        print(test_output)
        # part1 3653 too low
        # part1 3812, area calc was wrong

    elif part2:
        round_num = 0
        while True:
            round_num += 1
            all_elf_positions = set(map(lambda i: (i.x, i.y), elves))
            for elf in elves:
                elf.check_surroudings(all_elf_positions)
            all_targets = list(map(lambda i: i.target, elves))
            if len(set(all_targets)) != 1:
                for elf in elves:
                    elf.try_move(all_targets)
                choice_order = choice_order[1:] + choice_order[0]
            else:
                print(round_num)
                break


def day24():
    map_start = read_input("day24.txt")
    # initializing items
    start = (1, 0, 0)  # x, y, turn
    end = (len(map_start[0]) - 2, len(map_start) - 1)
    height = len(map_start)
    width = len(map_start[0])
    print(f"{width}, {height}")
    dirs = {">": [1, 0],
            "v": [0, 1],
            "<": [-1, 0],
            "^": [0, -1],
            }
    walls = set()
    all_storms = set()
    storms_positions = set()
    for y in range(height):
        for x in range(width):
            if map_start[y][x] == "#":
                walls.add((x, y))
            elif map_start[y][x] in "<>v^":
                all_storms.add((x, y, map_start[y][x]))
                storms_positions.add((x, y))

    def move_storm(storm):
        # takes storms of the form (x,y,direction of travel),
        # and reutrns a (new x, new y, same dir of travel)
        destination = (storm[0] + dirs[storm[2]][0], storm[1] + dirs[storm[2]][1], storm[2])
        if destination[0] == 0:
            destination = (width - 2, destination[1], destination[2])
        elif destination[0] == width - 1:
            destination = (1, destination[1], destination[2])
        elif destination[1] == 0:
            destination = (destination[0], height - 2, destination[2])
        elif destination[1] == height - 1:
            destination = (destination[0], 1, destination[2])
        return destination

    def print_status():
        line = ""
        for y in range(height):
            for x in range(width):
                if (x, y) in walls:
                    line += "#"
                elif (x, y) in storms_positions:
                    for storm in all_storms:
                        if (x, y) == (storm[0], storm[1]):
                            line += storm[2]
                            break
                else:
                    line += "."
            line += "\n"
        print(line)

    empty_spaces_each_turn = []
    for _ in range(1500):
        # build set of available spaces for each turn
        empty_spaces = {(1, 0), (width - 2, height - 1)}
        for y in range(1, height - 1):
            for x in range(1, width - 1):
                if (x, y) not in storms_positions:
                    empty_spaces.add((x, y))
        empty_spaces_each_turn.append(empty_spaces)

        # move all the storms
        new_storms = set()
        storms_positions = set()
        while len(all_storms) > 0:
            old_storm = all_storms.pop()
            new_storm = move_storm(old_storm)
            storms_positions.add((new_storm[0], new_storm[1]))
            new_storms.add(new_storm)
        all_storms = new_storms

    def find_vertices(pos_tuple):
        x, y, turn_number = pos_tuple
        next_map = empty_spaces_each_turn[turn_number + 1]
        possible_next_positions = set()
        for direction in dirs.values():
            if (x + direction[0], y + direction[1]) in next_map:
                possible_next_positions.add((x + direction[0], y + direction[1], turn_number + 1))
        if (x, y) in next_map:
            possible_next_positions.add((x, y, turn_number + 1))
        return possible_next_positions

    day24.decision_tree = {}
    day24.scores = {}

    def search(node):
        if node in day24.scores.keys():
            return day24.scores[node]
        if node[2] > start[2] + 450:
            day24.scores[node] = 9999
            return 9999
        if (node[0], node[1]) == end:
            print(node[2])
            return node[2]
        if node not in day24.decision_tree.keys():
            day24.decision_tree[node] = find_vertices(node)
        for next_node in day24.decision_tree[node]:
            result = search(next_node)
            if node not in day24.scores.keys():
                day24.scores[node] = result
            elif day24.scores[node] > result:
                day24.scores[node] = result
        if len(day24.decision_tree[node]) == 0:
            day24.scores[node] = 9999
        return day24.scores[node]

    part1 = search(start)
    print(f"part1: {part1}")
    start = (end[0], end[1], part1)
    end = (1, 0)
    day24.scores = {}
    day24.decision_tree = {}
    part2 = search(start)
    print(f"part2 (leg 1): {part2}")
    start = (1, 0, part2)
    end = end = (len(map_start[0]) - 2, len(map_start) - 1)
    day24.scores = {}
    day24.decision_tree = {}
    part2_leg2 = search(start)
    print(f"part2 (final_leg): {part2_leg2}")


def day25():
    """
    part 1 of day 25
    :return:
    """
    lines = read_input("day25.txt")
    vals = {"2": 2, "1": 1, "0": 0, "-": -1, "=": -2,
            2: '2', 1: "1", 0: '0', -1: "-", -2: "="}
    costs = []

    def snafu_to_int(snafu_str):
        dec = 0
        for i in range(len(snafu_str)):
            dec += vals[snafu_str[-(i + 1)]] * (5 ** i)
        return dec

    for line in lines:
        costs.append(snafu_to_int(line))
    print(sum(costs))

    def dec_to_snafu(decimal_int):
        digits = {"=": "-", "-": "0", "0": "1", "1": "2"}
        start = "1"
        next_start = start
        while snafu_to_int(next_start) < decimal_int:
            start = next_start
            if next_start[0] == "1" and len(next_start) > 1:
                next_start = "2" + next_start[1:]
            elif next_start == "1":
                next_start = "2"
            else:
                next_start = "1" + next_start[1:]
                next_start += "="

        i = 0
        while snafu_to_int(start) != decimal_int:
            min_next_digit = start[:i + 1] + "=" * len(start[i + 1:])
            max_next_digit = start[:i + 1] + "2" * len(start[i + 1:])
            if snafu_to_int(min_next_digit) == decimal_int:
                return min_next_digit
            if snafu_to_int(max_next_digit) == decimal_int:
                return max_next_digit
            if snafu_to_int(min_next_digit) <= decimal_int <= snafu_to_int(max_next_digit):
                print(min_next_digit)
                print(max_next_digit)
                i += 1
            else:
                start = list(start)
                start[i] = digits[start[i]]
                start = "".join(start)
        if snafu_to_int(start) == decimal_int:
            return start

    def test_snafu():
        lines = read_input("day25testSNAFU.txt")
        for line in lines[1:]:
            parts = line.strip().split(" ")
            snafu = parts[0]
            dec = parts[-1]
            print(f'converting: {dec_to_snafu(int(dec))}')
            print(f"correct answer: {snafu}")

    print(f"part 1: {dec_to_snafu(sum(costs))}")
    # attempt 1 -2-0=11=21-2201-01-2 # these attempts were built off of the bad observation that the digits lag 2 behind ish
    # attempt 2 -2-0=11=21-2201-0101
    # attempt 3 2-121-=10=200==2==21 #switching to binary search worked!
