import heapq
import time as timer

from helper_functions import *
from single_agent_planner import compute_heuristics, a_star, get_location, get_sum_of_cost


def detect_collision(path1, path2):
    ##############################
    # Task 3.1: Return the first collision that occurs between two robot paths (or None if there is no collision)
    #           There are two types of collisions: vertex collision and edge collision.
    #           A vertex collision occurs if both robots occupy the same location at the same timestep
    #           An edge collision occurs if the robots swap their location at the same timestep.
    #           You should use "get_location(path, t)" to get the location of a robot at time t.
    end_timestep = max(len(path1), len(path2))
    for timestep in range(end_timestep):
        path1_location = get_location(path1, timestep)
        path2_location = get_location(path2, timestep)
        if path1_location == path2_location:
            return [path1_location], timestep
        if timestep < end_timestep - 1:
            next_timestep = timestep + 1
            path1_location_next = get_location(path1, next_timestep)
            path2_location_next = get_location(path2, next_timestep)
            if path1_location == path2_location_next and path2_location == path1_location_next:
                return [path1_location, path2_location], next_timestep
    return None, None


def detect_collisions(paths):
    ##############################
    # Task 3.1: Return a list of first collisions between all robot pairs.
    #           A collision can be represented as dictionary that contains the id of the two robots, the vertex or edge
    #           causing the collision, and the timestep at which the collision occurred.
    #           You should use your detect_collision function to find a collision between two robots.
    num_of_agents = len(paths)
    collisions = []
    for agent1 in range(num_of_agents - 1):
        for agent2 in range(agent1 + 1, num_of_agents):
            collision_loc, coll_timestep = detect_collision(paths[agent1], paths[agent2])
            if collision_loc is not None and coll_timestep is not None:
                collisions.append({'a1': agent1, 'a2': agent2, 'loc': collision_loc, 'timestep': coll_timestep})
    return collisions


def standard_splitting(collision):
    ##############################
    # Task 3.2: Return a list of (two) constraints to resolve the given collision
    #           Vertex collision: the first constraint prevents the first agent to be at the specified location at the
    #                            specified timestep, and the second constraint prevents the second agent to be at the
    #                            specified location at the specified timestep.
    #           Edge collision: the first constraint prevents the first agent to traverse the specified edge at the
    #                          specified timestep, and the second constraint prevents the second agent to traverse the
    #                          specified edge at the specified timestep
    location = collision['loc']
    timestep = collision['timestep']
    constraints = []

    if len(location) == 1:
        constraints.append({'agent': collision['a1'], 'loc': location, 'timestep': timestep, 'positive': False})
        constraints.append({'agent': collision['a2'], 'loc': location, 'timestep': timestep, 'positive': False})
    else:
        constraints.append({'agent': collision['a1'], 'loc': location, 'timestep': timestep, 'positive': False})
        constraints.append({'agent': collision['a2'], 'loc': [location[1], location[0]], 'timestep': timestep, 'positive': False})

    return constraints


def disjoint_splitting(collision):
    ##############################
    # Task 4.1: Return a list of (two) constraints to resolve the given collision
    #           Vertex collision: the first constraint enforces one agent to be at the specified location at the
    #                            specified timestep, and the second constraint prevents the same agent to be at the
    #                            same location at the timestep.
    #           Edge collision: the first constraint enforces one agent to traverse the specified edge at the
    #                          specified timestep, and the second constraint prevents the same agent to traverse the
    #                          specified edge at the specified timestep
    #           Choose the agent randomly
    location = collision['loc']
    timestep = collision['timestep']
    constraints = []
    agent1 = collision['a1']
    agent2 = collision['a2']
    if random.randint(0, 1) == 0:
        constraints.append({'agent': agent1, 'loc': location, 'timestep': timestep, 'positive': True})
        constraints.append({'agent': agent1, 'loc': location, 'timestep': timestep, 'positive': False})
    elif len(location) == 1:
        constraints.append({'agent': agent2, 'loc': location, 'timestep': timestep, 'positive': True})
        constraints.append({'agent': agent2, 'loc': location, 'timestep': timestep, 'positive': False})
    else:
        constraints.append({'agent': agent2, 'loc': [location[1], location[0]], 'timestep': timestep, 'positive': True})
        constraints.append(
            {'agent': agent2, 'loc': [location[1], location[0]], 'timestep': timestep, 'positive': False})
    return constraints


def paths_violate_constraint(constraint, paths):
    assert constraint['positive'] is True
    rst = []
    for i in range(len(paths)):
        if i == constraint['agent']:
            continue
        curr = get_location(paths[i], constraint['timestep'])
        prev = get_location(paths[i], constraint['timestep'] - 1)
        if len(constraint['loc']) == 1:  # vertex constraint
            if constraint['loc'][0] == curr:
                rst.append(i)
        else:  # edge constraint
            if constraint['loc'][0] == prev or constraint['loc'][1] == curr \
                    or constraint['loc'] == [curr, prev]:
                rst.append(i)
    return rst


class CBSSolver(object):
    """The high-level search of CBS."""

    def __init__(self, my_map, starts, goals):
        """my_map   - list of lists specifying obstacle positions
        starts      - [(x1, y1), (x2, y2), ...] list of start locations
        goals       - [(x1, y1), (x2, y2), ...] list of goal locations
        """

        self.my_map = my_map
        self.starts = starts
        self.goals = goals
        self.num_of_agents = len(goals)

        self.num_of_generated = 0
        self.num_of_expanded = 0
        self.CPU_time = 0

        self.open_list = []

        # compute heuristics for the low-level search
        self.heuristics = []
        for goal in self.goals:
            self.heuristics.append(compute_heuristics(my_map, goal))

    def push_node(self, node):
        heapq.heappush(self.open_list, (node['cost'], len(node['collisions']), self.num_of_generated, node))
        print("Generate node {}".format(self.num_of_generated))
        self.num_of_generated += 1

    def pop_node(self):
        _, _, id, node = heapq.heappop(self.open_list)
        print("Expand node {}".format(id))
        self.num_of_expanded += 1
        return node

    def find_solution(self, disjoint=True):
        """ Finds paths for all agents from their start locations to their goal locations

        disjoint    - use disjoint splitting or not
        """

        self.start_time = timer.time()


        # Generate the root node
        # constraints   - list of constraints
        # paths         - list of paths, one for each agent
        #               [[(x11, y11), (x12, y12), ...], [(x21, y21), (x22, y22), ...], ...]
        # collisions     - list of collisions in paths
        root = {'cost': 0,
                'constraints': [],
                'paths': [],
                'collisions': []}
        for i in range(self.num_of_agents):  # Find initial path for each agent
            path = a_star(self.my_map, self.starts[i], self.goals[i], self.heuristics[i],
                          i, root['constraints'])
            if path is None:
                raise BaseException('No solutions')
            root['paths'].append(path)

        root['cost'] = get_sum_of_cost(root['paths'])
        root['collisions'] = detect_collisions(root['paths'])
        self.push_node(root)

        # # Task 3.1: Testing
        # print(root['collisions'])
        #
        # # Task 3.2: Testing
        # for collision in root['collisions']:
        #     print(standard_splitting(collision))

        ##############################
        # Task 3.3: High-Level Search
        #           Repeat the following as long as the open list is not empty:
        #             1. Get the next node from the open list (you can use self.pop_node()
        #             2. If this node has no collision, return solution
        #             3. Otherwise, choose the first collision and convert to a list of constraints (using your
        #                standard_splitting function). Add a new child node to your open list for each constraint
        #           Ensure to create a copy of any objects that your child nodes might inherit

        while len(self.open_list) > 0:
            P = self.pop_node()
            if len(P['collisions']) == 0:
                self.print_results(P)
                return P['paths']
            collision = P['collisions'][0]
            constraints = disjoint_splitting(collision) if disjoint else standard_splitting(collision)
            for constraint in constraints:
                Q = dict()
                Q['constraints'] = P['constraints'] + [constraint]
                Q['paths'] = copy.deepcopy(P['paths'])
                agent = constraint['agent']
                path = a_star(self.my_map, self.starts[agent], self.goals[agent],
                              self.heuristics[agent], agent, Q['constraints'])
                abandon = False
                if path is not None:
                    if constraint['positive']:
                        ids = paths_violate_constraint(constraint, P['paths'])
                        for other_agent in ids:
                            other_path = a_star(self.my_map, self.starts[other_agent], self.goals[other_agent],
                                                self.heuristics[other_agent], other_agent, Q['constraints'])
                            if other_path is not None:
                                Q['paths'][other_agent] = other_path
                            else:
                                abandon = True
                                break
                    Q['paths'][agent] = path
                    Q['collisions'] = detect_collisions(Q['paths'])
                    Q['cost'] = get_sum_of_cost(Q['paths'])
                    if not abandon:
                        self.push_node(Q)

        self.print_results(root)
        return root['paths']

    def print_results(self, node):
        print("\n Found a solution! \n")
        CPU_time = timer.time() - self.start_time
        print("CPU time (s):    {:.2f}".format(CPU_time))
        print("Sum of costs:    {}".format(get_sum_of_cost(node['paths'])))
        print("Expanded nodes:  {}".format(self.num_of_expanded))
        print("Generated nodes: {}".format(self.num_of_generated))


class ICBSSolver(object):
    """The high-level search of ICBS."""

    def __init__(self, my_map, starts, goals, discard_rate=0, heuristic_option=-1):
        """my_map   - list of lists specifying obstacle positions
        starts      - [(x1, y1), (x2, y2), ...] list of start locations
        goals       - [(x1, y1), (x2, y2), ...] list of goal locations
        """

        self.my_map = my_map
        self.starts = starts
        self.goals = goals
        self.num_of_agents = len(goals)
        self.heuristic_option = heuristic_option
        self.discard_rate = discard_rate
        self.num_of_generated = 0
        self.num_of_expanded = 0
        self.CPU_time = 0
        self.total_mdd_time = 0
        self.open_list = []

        # compute heuristics for the low-level search
        self.heuristics = []
        for goal in self.goals:
            self.heuristics.append(compute_heuristics(my_map, goal))

    def push_node(self, node):
        heapq.heappush(self.open_list,
                       (node['cost'] + node['h_value'], len(node['collisions']), self.num_of_generated, node))
        print("Generate node {}".format(self.num_of_generated))
        self.num_of_generated += 1

    def pop_node(self):
        _, _, id, node = heapq.heappop(self.open_list)
        print("Expand node {}".format(id))
        self.num_of_expanded += 1
        return node

    def find_solution(self, disjoint=True):
        """ Finds paths for all agents from their start locations to their goal locations

        disjoint    - use disjoint splitting or not
        """

        self.start_time = timer.time()

        root = {'cost': 0,
                'h_value': 0,
                'constraints': [],
                'paths': [],
                'collisions': [],
                'MDDs': []}
        for i in range(self.num_of_agents):  # Find initial path for each agent
            path = a_star(self.my_map, self.starts[i], self.goals[i], self.heuristics[i],
                          i, root['constraints'])
            if path is None:
                raise BaseException('No solutions')
            root['paths'].append(path)

        root['cost'] = get_sum_of_cost(root['paths'])
        root['collisions'] = detect_collisions(root['paths'])
        mdd_time = timer.time()
        for i in range(self.num_of_agents):
            # Construct the initial MDDs
            root['MDDs'].append(
                build_mdd(self.my_map, self.starts[i], self.goals[i], len(root['paths'][i]) - 1, self.heuristics[i],
                          root['constraints'], i, self.discard_rate))
        self.total_mdd_time = timer.time() - mdd_time
        self.push_node(root)

        while len(self.open_list) > 0:
            P = self.pop_node()
            if len(P['collisions']) == 0:
                self.print_results(P)
                return P['paths']
            # Optimize collision choice
            collision = return_optimal_conflict(P['collisions'], P['MDDs'])
            # collision = P['collisions'][0]
            constraints = disjoint_splitting(collision) if disjoint else standard_splitting(collision)
            for constraint in constraints:
                Q = dict()
                Q['constraints'] = P['constraints'] + [constraint]
                Q['paths'] = copy.deepcopy(P['paths'])
                agent = constraint['agent']
                path = a_star(self.my_map, self.starts[agent], self.goals[agent],
                              self.heuristics[agent], agent, Q['constraints'])
                abandon = False
                if path is not None:
                    if constraint['positive']:
                        ids = paths_violate_constraint(constraint, P['paths'])
                        for other_agent in ids:
                            other_path = a_star(self.my_map, self.starts[other_agent], self.goals[other_agent],
                                                self.heuristics[other_agent], other_agent, Q['constraints'])
                            if other_path is not None:
                                Q['paths'][other_agent] = other_path
                            else:
                                abandon = True
                                break
                    Q['paths'][agent] = path
                    Q['collisions'] = detect_collisions(Q['paths'])

                    Q['cost'] = get_sum_of_cost(Q['paths'])
                    Q['MDDs'] = copy.deepcopy(P['MDDs'])
                    # Reuse the MDDs from parent and only modify the MDD of the affected agent
                    mdd_time = timer.time()
                    Q['MDDs'][agent] = build_mdd(self.my_map, self.starts[agent], self.goals[agent],
                                                 len(Q['paths'][agent]) - 1,
                                                 self.heuristics[agent], Q['constraints'], agent, self.discard_rate)
                    self.total_mdd_time += timer.time() - mdd_time
                    if not abandon:
                        if self.heuristic_option == 0:
                            Q['h_value'] = compute_cg_heuristic(Q['MDDs'], self.num_of_agents)
                        elif self.heuristic_option == 1:
                            Q['h_value'] = compute_dg_heuristic(Q['MDDs'], self.num_of_agents)
                        else:
                            Q['h_value'] = 0
                        self.push_node(Q)

        self.print_results(root)
        return root['paths']

    def print_results(self, node):
        print("\n Found a solution! \n")
        CPU_time = timer.time() - self.start_time
        print("Sum of costs:    {}".format(get_sum_of_cost(node['paths'])))
        print("Expanded nodes:  {}".format(self.num_of_expanded))
        print("Generated nodes: {}".format(self.num_of_generated))
        print("CBS time (s):    {:.2f}".format(CPU_time))
        print("MDD constructing time (s): {:.2f}".format(self.total_mdd_time))
        print("Total time (s): {:.2f}".format(self.total_mdd_time + CPU_time))
