import collections
import copy
import random

import networkx
from networkx.algorithms.approximation import vertex_cover

from single_agent_planner import move, is_constrained, build_constraint_table


def build_mdd(my_map, start_loc, goal_loc, cost, h_values_table, constraints, agent, discard_rate, prev_mdd=None):
    # Build the constraints table
    constraints_table = build_constraint_table(constraints, agent)

    # Reuse the parent's MDD
    if prev_mdd:
        new_mdd = copy.deepcopy(prev_mdd)
        # The new constraints only affect the MDD levels below the lowest timestep (depth) of the new constraints
        # with the lowest timestep (depth)
        constraint_with_lowest_ts = constraints[0]
        for constraint in constraints:
            if constraint['timestep'] < constraint_with_lowest_ts['timestep']:
                constraint_with_lowest_ts = constraint
        # Locate the level
        level = constraint_with_lowest_ts['timestep']
        # Locate the conflict position (node)
        parent = None
        for node in prev_mdd[level]:
            parent = node['prev']
            if is_constrained(parent['curr_loc'], node['curr_loc'], level, constraints_table):
                # Cut off all nodes below the conflict position's parent node and the reproduce the MDDs
                lower_mdd_dict = build_mdd(my_map, parent, goal_loc, cost - level, h_values_table, constraints, agent,
                                           discard_rate)
                break
        # Merge the lower mdd to the upper mdd

    root_node = {'prev': None, 'cost': 0, 'curr_loc': start_loc}
    # Build open_list as a queue
    open_list = collections.deque([root_node])
    # Build MDD as a hash table, each entry is a list consist of all possible locations
    mdd_dict = dict()
    mdd_add_dict = dict()

    for i in range(0, cost + 1):
        mdd_dict[i] = []
    while len(open_list) > 0:
        current_node = open_list.popleft()
        # Reach the bottom of MDD
        if current_node['curr_loc'] == goal_loc and current_node['cost'] == cost:

            index = cost
            curr = current_node
            # Fill the MDD hash table
            child = {'curr_loc': None}
            while curr is not None:
                if curr['curr_loc'] not in mdd_dict[index]:
                    mdd_dict[index].append(curr['curr_loc'])
                    mdd_add_dict[(index, curr['curr_loc'])] = [child['curr_loc']]
                else:
                    mdd_add_dict[(index, curr['curr_loc'])].append(child['curr_loc'])
                child = curr
                curr = curr['prev']
                index -= 1

            continue
        for direction in range(5):
            next_location = move(current_node['curr_loc'], direction)
            # Reach a block / boundary / constraint location
            if next_location[0] < 0 or \
                    next_location[0] >= len(my_map) or next_location[1] < 0 or \
                    next_location[1] >= len(my_map[0]) or \
                    my_map[next_location[0]][next_location[1]] or \
                    is_constrained(current_node['curr_loc'], next_location, current_node['cost'] + 1,
                                   constraints_table):
                continue
            child_node = {'prev': current_node, 'cost': current_node['cost'] + 1, 'curr_loc': next_location}
            # If the f value larger than the cost, it is impossible to include such location into the path
            if child_node['cost'] + h_values_table[next_location] <= cost:
                if random.random() >= discard_rate:
                    open_list.append(child_node)
    return mdd_dict, mdd_add_dict


def return_optimal_conflict(conflicts, mdds):
    semi_cardinals = []
    non_cardinals = []
    for conflict in conflicts:
        mdd1 = mdds[conflict['a1']][0]
        mdd2 = mdds[conflict['a2']][0]
        location = conflict['loc']
        timestep = conflict['timestep']

        # One agent already reaches its goal
        if (len(mdd1.keys()) - 1 < timestep or len(mdd2.keys()) - 1 < timestep):
            non_cardinals.append(conflict)
            continue
        # Cardinal conflict
        if len(mdd1[timestep]) == 1 and len(mdd2[timestep]) == 1 and mdd1[timestep][0] == location and mdd2[timestep][
            0] == location:
            return conflict
        # Semi-cardinal conflict
        if (len(mdd1[timestep]) == 1 or len(mdd2[timestep]) == 1) and location in mdd1[timestep] and location in mdd2[
            timestep]:
            semi_cardinals.append(conflict)
        # Non cardinal conflict
        else:
            non_cardinals.append(conflict)
    if len(semi_cardinals) > 0:
        return semi_cardinals[0]
    return non_cardinals[0]


def compute_cg_heuristic(MDDs, agents):
    conflict_graph = networkx.Graph(name="Conflict Graph")
    for i in range(agents - 1):
        for j in range(i + 1, agents):
            mdd1 = MDDs[i][0]
            mdd2 = MDDs[j][0]
            depth = min(len(mdd1.keys()), len(mdd2.keys()))
            for d in range(1, depth):
                if len(mdd1[d]) == 1 and len(mdd2[d]) == 1 and mdd1[d][0] == mdd2[d][0]:
                    # node_i = {j: 1}
                    # node_j = {i: 1}
                    # conflict_graph[i] = node_i
                    # conflict_graph[j] = node_j
                    conflict_graph.add_node(i)
                    conflict_graph.add_node(j)
                    conflict_graph.add_edge(i, j)
    cg_h_value = len(vertex_cover.min_weighted_vertex_cover(conflict_graph))
    return cg_h_value


def join_MDD(mdd1, mdd2):
    basic_mdd1 = mdd1[0]
    basic_mdd2 = mdd2[0]
    additional_mdd1 = mdd1[1]
    additional_mdd2 = mdd2[1]
    len_mdd1 = len(basic_mdd1.keys())
    len_mdd2 = len(basic_mdd2.keys())

    max_len = max(len_mdd1, len_mdd2)
    # Inconsistent depth, add dummy nodes
    basic_mdd1 = copy.deepcopy(basic_mdd1)
    basic_mdd2 = copy.deepcopy(basic_mdd2)
    joint_MDD = {}
    if len_mdd1 < len_mdd2:
        for i in range(len_mdd1, len_mdd2):
            basic_mdd1[i] = [basic_mdd1[len_mdd1 - 1]]
    elif len_mdd2 < len_mdd1:
        for i in range(len_mdd2, len_mdd1):
            basic_mdd2[i] = [basic_mdd2[len_mdd2 - 1]]

    for i in range(1, max_len):
        joint_MDD[i] = []

    root1 = basic_mdd1[0][0]
    root2 = basic_mdd2[0][0]
    joint_MDD[0] = [(root1, root2)]
    for i in range(1, max_len):
        for node in joint_MDD[i - 1]:

            for child1 in additional_mdd1[(i - 1, node[0])]:
                for child2 in additional_mdd2[(i - 1, node[1])]:
                    if child1 is None or child2 is None:
                        continue
                    if child1 == child2 and i != max_len - 2:
                        continue
                    joint_node = (child1, child2)
                    if joint_node not in joint_MDD[i]:
                        joint_MDD[i].append(joint_node)
    return joint_MDD, max_len


def compute_dg_heuristic(MDDs, agents):
    gd_graph = networkx.Graph(name="Pairwise Dependency Graph")
    for i in range(agents - 1):
        for j in range(i + 1, agents):
            mdd1 = MDDs[i]
            mdd2 = MDDs[j]
            joint_mdd, max_len = join_MDD(mdd1, mdd2)
            # Two agents are dependent
            if len(joint_mdd[max_len - 1]) == 0:
                gd_graph.add_node(i)
                gd_graph.add_node(j)
                gd_graph.add_edge(i, j)
    dg_h_value = len(vertex_cover.min_weighted_vertex_cover(gd_graph))
    return dg_h_value
