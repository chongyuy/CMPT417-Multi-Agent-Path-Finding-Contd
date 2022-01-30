import time as timer
from single_agent_planner import compute_heuristics, a_star, get_sum_of_cost


class PrioritizedPlanningSolver(object):
    """A planner that plans for each robot sequentially."""

    def __init__(self, my_map, starts, goals):
        """my_map   - list of lists specifying obstacle positions
        starts      - [(x1, y1), (x2, y2), ...] list of start locations
        goals       - [(x1, y1), (x2, y2), ...] list of goal locations
        """

        self.my_map = my_map
        self.starts = starts
        self.goals = goals
        self.num_of_agents = len(goals)

        self.CPU_time = 0

        # compute heuristics for the low-level search
        self.heuristics = []
        for goal in self.goals:
            self.heuristics.append(compute_heuristics(my_map, goal))

    def find_solution(self):
        """ Finds paths for all agents from their start locations to their goal locations."""
        start_time = timer.time()
        result = []
        constraints = []

        map_size = 0
        for row in self.my_map:
            for col in row:
                if not col:
                    map_size += 1

        # constraints = [{'agent': 0, 'loc': [(1, 5)], 'timestep': 4, 'positive': False},
        #                {'agent': 1, 'loc': [(1, 2), (1, 3)], 'timestep': 1, 'positive': False}]

        # constraints = [{'agent': 0, 'loc': [(1, 5)], 'timestep': 10, 'positive': False}]

        # constraints = [{'agent': 1, 'loc': [(1, 2)], 'timestep': 2, 'positive': False},
        #                {'agent': 1, 'loc': [(1, 3)], 'timestep': 2, 'positive': False},
        #                {'agent': 1, 'loc': [(1, 4)], 'timestep': 2, 'positive': False}]
        for i in range(self.num_of_agents):  # Find path for each agent
            path = a_star(self.my_map, self.starts[i], self.goals[i], self.heuristics[i],
                          i, constraints)
            all_paths_len = 0
            for temp_path in result:
                all_paths_len += len(temp_path)
            upper_bound = map_size + all_paths_len
            if path is None or len(path) >= upper_bound:
                raise BaseException('No solutions')
            result.append(path)


            ##############################
            # Task 2: Add constraints here
            #         Useful variables:
            #            * path contains the solution path of the current (i'th) agent, e.g., [(1,1),(1,2),(1,3)]
            #            * self.num_of_agents has the number of total agents
            #            * constraints: array of constraints to consider for future A* searches
            ##############################

            for location_i in range(len(path)):
                for agent in range(i + 1, self.num_of_agents):
                    # Task 2.1
                    constraints.append({'agent': agent, 'loc': [path[location_i]], 'timestep': location_i, 'positive': False})
                    if location_i < len(path) - 1:
                        # Task 2.2
                        constraints.append({'agent': agent, 'loc': [path[location_i + 1], path[location_i]], 'timestep': location_i + 1, 'positive': False})

            for agent in range(i + 1, self.num_of_agents):
                # Additional constraints
                for time in range(len(path), upper_bound + 1):
                    constraints.append({'agent': agent, 'loc': [path[-1]], 'timestep': time, 'positive': False})
        self.CPU_time = timer.time() - start_time

        print("\n Found a solution! \n")
        print("CPU time (s):    {:.2f}".format(self.CPU_time))
        print("Sum of costs:    {}".format(get_sum_of_cost(result)))
        print(result)
        return result
