import os
import random


def generate_maps(height, width, dense_fac, agents, num_of_maps):
    folder = os.path.exists('test_maps')
    if not folder:
        os.makedirs('test_maps')
    for i in range(num_of_maps):
        map = []
        file_name = 'test' + str(i) + '.txt'

        if os.path.exists('test_maps/' + file_name):
            os.remove('test_maps/' + file_name)
        file_obj = open('test_maps/' + file_name, 'w')
        # set the size of map to given height and width
        file_obj.writelines(str(height) + ' ' + str(width))
        file_obj.write('\n')
        # randomly generate the map
        for j in range(height):
            map.append([])
            line = ''
            for k in range(width):
                rand_num = random.randint(0, dense_fac)
                if rand_num == 0:
                    map[j].append(0)
                    line = line + '@' + ' '
                else:
                    map[j].append(1)
                    line = line + '.' + ' '
            file_obj.writelines(line)
            file_obj.write('\n')
        # randomly generate agents on maps
        file_obj.writelines(str(agents))
        file_obj.write('\n')
        for agent in range(agents):
            while True:
                start_height = random.randint(0, height - 1)
                start_width = random.randint(0, width - 1)
                if map[start_height][start_width] == 1:
                    map[start_height][start_width] = 0
                    break
            while True:
                goal_height = random.randint(0, height - 1)
                goal_width = random.randint(0, width - 1)
                if map[goal_height][goal_width] == 1:
                    map[goal_height][goal_width] = 0
                    break
            file_obj.writelines(
                str(start_height) + ' ' + str(start_width) + ' ' + str(goal_height) + ' ' + str(goal_width))
            file_obj.write('\n')
        file_obj.close()


generate_maps(8, 8, 1, 4, 20)
