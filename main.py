"""
Karo Zhu
CSDS 391
Programming Assignment 1
A* Search
"""
from enum import Enum, auto
import math
import random
import time

random.seed(391)
class Search():
    solved_state = "b12 345 678"
    max_default = 1000

    def __init__(self, size):
        self.current_node = self.Node(Search.solved_state, self, None, None)
        self.goal_state = Search.solved_state
        self.max_nodes = Search.max_default
        self.curr_nodes = 0

        #used for A*
        self.tree = dict()

        #used for local beam search
        self.k_states = set()
        self.successors = set()

        self.size = size
        self.lefts = list()
        self.rights = list()
        left_in = 0
        right_in = size - 1
        while left_in < math.pow(size, 2) + size - 1:
            self.lefts.append(left_in)
            self.rights.append(right_in)
            left_in += size + 1
            right_in += size + 1

    # heuristic should be h1 or h2 (3.6 in book)
    # when goal is found, code prints number of tile moves needed to obtain the solution and
    # the sequence of moves from starting state to goal state
    def solve_a_star(self, heuristic):
        self.current_node.randomize_state(30) #randomized by an arbitrary number
        self.tree = dict()
        self.tree[self.current_node] = self.expand_current_node(heuristic) #expands the root
        while self.current_node.current_state != Search.solved_state:
            lowest_cost_node, lowest_cost = self.find_lowest_cost()
            self.current_node = lowest_cost_node
            path, _ = self.find_path(lowest_cost_node)
            path.reverse()
            ptr_dict = self.tree
            for index in range(0, len(path)):
                ptr_dict = ptr_dict[path[index]]
            ptr_dict[self.current_node] = self.expand_current_node(heuristic)

            if self.curr_nodes >= self.max_nodes:
                raise Exception("Maximum number of nodes to consider has been reached. Solution unknown")

        path, dirs = self.find_path(self.current_node)
        dirs.reverse()
        return len(path), dirs

    #given a node, trace the path back to the root of the tree, including the directions traversed
    def find_path(self, node):
        ptr_node = node
        path = list()
        dirs = list()
        while ptr_node.parent_node is not None and ptr_node.last_move is not None:
            path.append(ptr_node.parent_node)
            dirs.append(ptr_node.last_move)
            ptr_node = ptr_node.parent_node
        return path, dirs

    #helper function to find the lowest cost path in the tree so far
    def find_lowest_cost(self):
        cost_dict = self.find_lowest_cost_dict(self.tree, 1)
        min_item = list(cost_dict.keys())[0]
        min_value = cost_dict[min_item]
        for item in cost_dict.keys():
            if cost_dict[item] < min_value:
                min_value = cost_dict[item]
                min_item = item
        return min_item, min_value

    #recursive helper method for find_lowest_cost to flatten the dictionary
    def find_lowest_cost_dict(self, in_dict, layer):
        cost_dict = dict()
        for node in in_dict.keys():
            if isinstance(in_dict[node], dict):
                cost_dict.update(self.find_lowest_cost_dict(in_dict[node], layer + 1))
            else:
                cost_dict[node] = in_dict[node] + layer
        return cost_dict

    #helper method to expand the current node
    def expand_current_node(self, heuristic):
        self.current_node.get_possible_actions()
        possible_actions = self.current_node.possible_actions
        node_cost_dict = dict()
        for action in possible_actions:
            new_node = self.Node(self.current_node.current_state, self, action, self.current_node)
            new_node.move(action)
            if heuristic == "h1":
                node_cost_dict[new_node] = new_node.h_1()
            elif heuristic == "h2":
                node_cost_dict[new_node] = new_node.h_2()
            else:
                raise Exception("Invalid heuristic input")
        self.curr_nodes += len(node_cost_dict)
        return node_cost_dict

    # solve puzzle using local beam search with k states
    # need to define an evaluation function, minimum of zero at the goal state
    def solve_beam(self, k):
        self.current_node.randomize_state(25) #Start at a random, solvable puzzle
        self.k_states.clear()
        self.successors.clear()
        for in_i in range(0, k):
            new_node = self.Node(self.current_node.current_state, self, self.current_node, None)
            new_node.randomize_state(random.randint(20,30))
            self.k_states.add(new_node)
        self.curr_nodes += len(self.k_states)
        solution_node = self.current_node
        while solution_node.current_state != Search.solved_state:
            for node_i in self.k_states:
                node_i.get_possible_actions()
                actions = node_i.possible_actions
                for action in actions:
                    new_node = self.Node(node_i.current_state, self, action, node_i)
                    new_node.move(action)
                    self.successors.add(new_node)
            self.k_states.clear()

            self.curr_nodes += len(self.successors)
            self.k_states.update(self.find_k_best(k))
            self.curr_nodes += len(self.k_states)

            self.successors.clear()
            for node_j in self.k_states:
                if node_j.current_state == Search.solved_state:
                    solution_node = node_j

            if self.curr_nodes >= self.max_nodes:
                raise Exception("Maximum number of nodes to consider has been reached. Solution unknown")
        path, dirs = self.find_path(solution_node)
        dirs.reverse()
        return len(path), dirs, solution_node.current_state

    def find_k_best(self, k):
        k_best_set = set()
        eval_dict = dict()
        for state in self.successors:
            eval_dict[state] = state.h_2()

        i = 0
        for e in sorted(eval_dict.items(), key=lambda i : i[1]):
            if i < k:
                k_best_set.add(e[0])
                i += 1
            else:
                break
        return k_best_set

    def generate_random_state(self):
        rand_state = ""
        num_list = list(range(1, int(math.pow(self.size, 2))))
        random.shuffle(num_list)
        in_j = 0
        b_location = self.generate_valid_b_location()
        for in_i in range(0, len(self.current_node.current_state)):
            if (in_i - 1) in self.rights:
                rand_state += " "
            elif in_i == b_location:
                rand_state += "b"
            else:
                rand_state += str(num_list[in_j])
                in_j += 1
        return rand_state

    def generate_valid_b_location(self):
        b_location = random.randint(0, len(self.current_node.current_state) - 1)
        if (b_location - 1) in self.rights:
            return self.generate_valid_b_location()
        return b_location

    def set_max_nodes(self, n):
        self.max_nodes = n

    class Node:
        def __init__(self, initial_state, search, last_move, parent_node):
            self.current_state = initial_state
            self.search = search
            self.possible_actions = set()
            self.last_move = last_move
            self.parent_node = parent_node

        #move the blank tile in a certain direction
        def move(self, direction):
            b_location = self.current_state.index("b")
            curStr = self.current_state
            if direction == Direction.UP:
                swap_value_index = b_location - (self.search.size + 1)
                if self.can_move_direction(Direction.UP):
                    self.current_state = curStr[:swap_value_index] + "b" \
                                         + curStr[swap_value_index + 1:b_location] + curStr[swap_value_index] \
                                         + curStr[b_location + 1:]
            elif direction == Direction.DOWN:
                swap_value_index = b_location + (self.search.size + 1)
                if self.can_move_direction(Direction.DOWN):
                    self.current_state = curStr[:b_location] + curStr[swap_value_index] \
                                         + curStr[b_location + 1:swap_value_index] + "b" \
                                         + curStr[swap_value_index + 1:]
            elif direction == Direction.LEFT:
                swap_value_index = b_location - 1
                if self.can_move_direction(Direction.LEFT):
                    self.current_state = curStr[:swap_value_index] + "b" + curStr[swap_value_index] \
                                         + curStr[swap_value_index + 2:]
            elif direction == Direction.RIGHT:
                swap_value_index = b_location + 1
                if self.can_move_direction(Direction.RIGHT):
                    self.current_state = curStr[:b_location] + curStr[swap_value_index] + "b" \
                                         + curStr[b_location + 2:]
            else:
                raise Exception("Invalid direction input")
            return self.current_state

        # Make n random moves from the goal state
        # Ensures the puzzle is solvable
        def randomize_state(self, n):
            for i in range(0, n):
                self.move(Direction.get_rand_direction())

        #returns all possible actions from the current state
        def get_possible_actions(self):
            poss_actions = set()
            for direction in Direction:
                if self.can_move_direction(direction):
                    poss_actions.add(direction)
            self.possible_actions = poss_actions

        #returns true or false value depending on whether the blank tile can be moved in the input direction
        def can_move_direction(self, direction):
            b_location = self.current_state.index("b")
            if direction == Direction.UP:
                return b_location >= self.search.size
            elif direction == Direction.DOWN:
                return b_location < len(self.current_state) - (self.search.size + 1)
            elif direction == Direction.LEFT:
                return b_location not in self.search.lefts
            elif direction == Direction.RIGHT:
                return b_location not in self.search.rights
            else:
                raise Exception("Invalid direction input")

        #function to calculate h_1 value of current state
        #number of displaced tiles
        def h_1(self):
            num_displaced_tiles = 0
            for num in range(0, len(self.current_state)):
                if Search.solved_state[num] != self.current_state[num]:
                    num_displaced_tiles += 1
            return num_displaced_tiles

        #function to calculate h_2 value of current state
        #sum of all Manhattan distances of each tile to their respective original locations
        def h_2(self):
            curr_sum = 0
            for num in range(0, len(self.current_state)):
                if num - 1 not in self.search.rights and self.current_state[num] != "b":
                    current_num = int(self.current_state[num])
                    current_num_actual_position = math.floor(current_num / self.search.size) * (self.search.size + 1) + (
                                current_num % self.search.size)

                    horiz_dist = abs((current_num_actual_position % (self.search.size + 1)) - (num % (self.search.size + 1)))
                    vert_dist = abs(
                        math.floor(current_num_actual_position / (self.search.size + 1)) - math.floor(num / (self.search.size + 1)))
                    curr_sum += (horiz_dist + vert_dist)
            return curr_sum

        def set_state(self, state):
            self.current_state = state

        def print_state(self):
            print(self.current_state)

        def __str__(self):
            return self.current_state


class Direction(Enum):
    UP = auto()
    DOWN = auto()
    LEFT = auto()
    RIGHT = auto()

    @classmethod
    def get_rand_direction(cls):
        return Direction(random.randint(1,4))

def main():
    inputFile = "input.txt"
    with open(inputFile, "r+") as file:
        search = Search(3)
        line = file.readline()
        while line != "":
            if "setState" in line:
                state = line[line.index(" ") + 1:len(line) - 1]
                search.current_node.set_state(state)
            elif "printState" in line:
                search.current_node.print_state()
                print_str = search.current_node.current_state
                file.write(print_str)
            elif "move" in line:
                dir_str = line.split(" ")[1]
                direction = None
                if "UP" in dir_str.upper():
                    direction = Direction.UP
                elif "DOWN" in dir_str.upper():
                    direction = Direction.DOWN
                elif "LEFT" in dir_str.upper():
                    direction = Direction.LEFT
                elif "RIGHT" in dir_str.upper():
                    direction = Direction.RIGHT
                else:
                    raise Exception("invalid direction input")
                search.current_node.move(direction)
            elif "randomizeState" in line:
                search.current_node.randomize_state(int(line.split(" ")[1]))
            elif "solve A-star" in line:
                heuristic = ""
                if "h1" in line:
                    heuristic = "h1"
                elif "h2" in line:
                    heuristic = "h2"
                else:
                    raise Exception("Invalid heuristic")
                solution = str(search.solve_a_star(heuristic))
                file.write(solution)
            elif "solve beam" in line:
                solution = str(search.solve_beam(int(line[line.index("m")+2: len(line) - 1])))
                file.write(solution)
            elif "maxNodes" in line:
                search.set_max_nodes(int(line.split(" ")[1]))
            else:
                raise Exception("Parsing error")
            line = file.readline()

def exp_a(max_nodes):
    random_puzzles = list()
    for ind in range(0, 50):
        search = Search(3)
        search.current_node.randomize_state(30)
        random_puzzles.append(search.current_node.current_state)

    unsolved = 0
    for puzzle in random_puzzles:
        try:
            search_a = Search(3)
            search_a.max_nodes = max_nodes
            search_a.current_node.current_state = puzzle
            search_a.solve_a_star("h2")
        except:
            unsolved += 1

    return unsolved / 50

def exp_b():
    times_h1 = list()
    for ind in range(0, 100):
        search = Search(3)
        start_time = time.time_ns()
        search.solve_a_star("h1")
        stop_time = time.time_ns()
        times_h1.append(stop_time - start_time)

    times_h2 = list()
    for ind in range(0, 100):
        search = Search(3)
        start_time = time.time_ns()
        search.solve_a_star("h2")
        stop_time = time.time_ns()
        times_h2.append(stop_time - start_time)

    average_time_h1 = sum(times_h1) / len(times_h1)
    average_time_h2 = sum(times_h2) / len(times_h2)
    return average_time_h1, average_time_h2

def exp_c():
    h1_lengths = list()
    for ind in range(0, 100):
        search = Search(3)
        solution, _ = search.solve_a_star("h1")
        h1_lengths.append(solution)

    h2_lengths = list()
    for ind in range(0, 100):
        search = Search(3)
        solution, _ = search.solve_a_star("h2")
        h2_lengths.append(solution)

    beam_lengths = list()
    try:
        for ind in range(0, 100):
            search = Search(3)
            solution, _ = search.solve_beam(2)
            beam_lengths.append(solution)
    except:
        beam_average = sum(beam_lengths) / len(beam_lengths)
        print(beam_average)
    finally:
        h1_average = sum(h1_lengths) / len(h1_lengths)
        h2_average = sum(h2_lengths) / len(h2_lengths)
        return h1_average, h2_average, beam_average

def exp_d():
    random_puzzles = list()
    for ind in range(0, 50):
        search = Search(3)
        search.current_node.randomize_state(30)
        random_puzzles.append(search.current_node.current_state)

    unsolved_h1 = 0
    for puzzle in random_puzzles:
        try:
            search_a = Search(3)
            search_a.current_node.current_state = puzzle
            search_a.solve_a_star("h1")
        except:
            unsolved_h1 += 1

    unsolved_h2 = 0
    for puzzle in random_puzzles:
        try:
            search_a = Search(3)
            search_a.current_node.current_state = puzzle
            search_a.solve_a_star("h2")
        except:
            unsolved_h2 += 1

    unsolved_beam = 0
    for puzzle in random_puzzles:
        try:
            search_a = Search(3)
            search_a.max_nodes = 100000
            search_a.current_node.current_state = puzzle
            search_a.solve_beam(2)
        except:
            unsolved_beam += 1

    return unsolved_h1 / 50, unsolved_h2 / 50, unsolved_beam / 50

main()
