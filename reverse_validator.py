import numpy as np
from itertools import combinations
import copy

row = 10
col = 10

upper_limit = row - 1
#
# prob_dens = np.array([[0] * col for i in range(row)])

# prob_dens[0] = [7, 7, 7, 7, 7, 5, 7, 7, 2, 7]
# prob_dens[1] = [0, 7, 4, 7, 7, 5, 7, 7, 2, 7]
# prob_dens[2] = [7, 7, 4, 7, 7, 5, 7, 7, 7, 7]
# prob_dens[3] = [0, 7, 4, 7, 7, 5, 7, 7, 2, 2]
# prob_dens[4] = [7, 7, 4, 7, 7, 5, 7, 7, 7, 7]
# prob_dens[5] = [7, 7, 7, 7, 7, 7, 7, 3, 7, 3]
# prob_dens[6] = [7, 0, 7, 7, 2, 7, 7, 3, 7, 3]
# prob_dens[7] = [7, 7, 0, 7, 2, 7, 7, 3, 7, 3]
# prob_dens[8] = [7, 7, 0, 7, 7, 7, 7, 7, 7, 7]
# prob_dens[9] = [0, 0, 0, 7, 7, 0, 7, 7, 0, 7]

# board = prob_dens
seg_2_fields = []
seg_3_fields = []
seg_4_fields = []
seg_5_fields = []
forbidden = []
all_ships = []
n_seg5 = 1
n_seg4 = 1
n_seg3 = 2
n_seg2 = 3

def validate (board):
    def count_fields(board):
        for x in range(10):
            for y in range(10):
                # print(x,y)
                if board[x][y] == 2:
                    seg_2_fields.append([x, y])
                if board[x][y] == 3:
                    seg_3_fields.append([x, y])
                if board[x][y] == 4:
                    seg_4_fields.append([x, y])
                if board[x][y] == 5:
                    seg_5_fields.append([x, y])

        return seg_2_fields, seg_3_fields, seg_4_fields, seg_5_fields

    def checkConsecutive(lis):
        return sorted(lis) == list(range(min(lis), max(lis) + 1))

    def distance(field1, field2):
        d = ((field1[0] - field2[0]) ** 2 + (field1[1] - field2[1]) ** 2) ** 0.5
        return d

    def group_2segs(seg_2_fields):
        temp_lis = seg_2_fields
        first_ship = []
        second_ship = []
        third_ship = []
        comb = combinations(seg_2_fields, 2)
        counter = 0
        seg2_ships = []
        for i in comb:
            if distance(i[0], i[1]) == 1:
                seg2_ships.append(i)
        if len(seg2_ships) == 3:
            all_ships.append(seg2_ships[0])
            all_ships.append(seg2_ships[1])
            all_ships.append(seg2_ships[2])
            return True
        else:
            return False

    def group_3segs(seg_3_fields):
        # for field1 in seg_3_fields:
        #     if field1 not in temp_ship:
        #         for field1 in seg_3_fields:
        first_ship = []
        second_ship = []
        comb = combinations(seg_3_fields, 3)
        for i in list(comb):
            result = False
            x0 = i[0][0]
            x1 = i[1][0]
            x2 = i[2][0]
            y0 = i[0][1]
            y1 = i[1][1]
            y2 = i[2][1]
            if x0 == x1 and x0 == x2:
                lis = [y0, y1, y2]
                result = checkConsecutive(lis)
            elif y0 == y1 and y0 == y2:
                lis = [x0, x1, x2]
                result = checkConsecutive(lis)
            if result == True:
                first_ship = i
                break
        for k in seg_3_fields:
            if k not in first_ship:
                second_ship.append(k)
        # # Check if the second ship contains consecutive fields
        result2 = False
        x0 = second_ship[0][0]
        x1 = second_ship[1][0]
        x2 = second_ship[2][0]
        y0 = second_ship[0][1]
        y1 = second_ship[1][1]
        y2 = second_ship[2][1]
        if x0 == x1 and x0 == x2:
            lis = [y0, y1, y2]
            result2 = checkConsecutive(lis)
        elif y0 == y1 and y0 == y2:
            lis = [x0, x1, x2]
            result2 = checkConsecutive(lis)
        # print('first ship is: ', first_ship)
        # print('second ship is: ', second_ship)
        all_ships.append(first_ship)
        all_ships.append(second_ship)
        return result2

    def group_4segs(seg_4_fields):
        result = False
        i = seg_4_fields
        x0 = i[0][0]
        x1 = i[1][0]
        x2 = i[2][0]
        x3 = i[3][0]
        y0 = i[0][1]
        y1 = i[1][1]
        y2 = i[2][1]
        y3 = i[3][1]
        if x0 == x1 and x0 == x2:
            lis = [y0, y1, y2, y3]
            result = checkConsecutive(lis)
        elif y0 == y1 and y0 == y2:
            lis = [x0, x1, x2, x3]
            result = checkConsecutive(lis)
        if result == True:
            all_ships.append(seg_4_fields)
        return result

    def group_5segs(seg_5_fields):
        result = False
        i = seg_5_fields
        x0 = i[0][0]
        x1 = i[1][0]
        x2 = i[2][0]
        x3 = i[3][0]
        x4 = i[4][0]
        y0 = i[0][1]
        y1 = i[1][1]
        y2 = i[2][1]
        y3 = i[3][1]
        y4 = i[4][1]
        if x0 == x1 and x0 == x2:
            lis = [y0, y1, y2, y3, y4]
            result = checkConsecutive(lis)
        elif y0 == y1 and y0 == y2:
            lis = [x0, x1, x2, x3, x4]
            result = checkConsecutive(lis)
        if result == True:
            all_ships.append(seg_5_fields)
        return result

    def forbid_diagonals(board, curr_ship):
        for field in curr_ship:
            x = int(field[0])
            y = int(field[1])
            # Add adjacent diagonal fields to the forbidden fields' list
            if x > 0:
                if y > 0:
                    forbidden.append([x - 1, y - 1])  # Upper-left
            if x < upper_limit:
                if y < upper_limit:
                    forbidden.append([x + 1, y + 1])  # Lower-right
            if y > 0:
                if x < upper_limit:
                    forbidden.append([x + 1, y - 1])  # Lower-left
            if y < upper_limit:
                if x > 0:
                    forbidden.append([x - 1, y + 1])  # Upper-right
        return forbidden

    def identify_orientation(curr_ship):
        # If x coordinate of first 2 fields in a ship are equal, this means it's horizontal
        if curr_ship[0][0] == curr_ship[1][0]:
            return "hor"
        # If y coordinate of first 2 fields in a ship are equal, this means it's vertical
        elif curr_ship[0][1] == curr_ship[1][1]:
            return "ver"
        else:
            return "error"

    def forbid_endings(board, curr_ship, orient, upper_limit):
        if orient == "hor":
            # Set x as constant
            x = int(curr_ship[0][0])
            # Find minimum column number as the left end of the ship
            left_end = int(np.min(curr_ship[:, 1]))
            # print('left_end is : ', left_end)
            if left_end > 0:
                forbidden.append([x, left_end - 1])
            # Find maximum column number as the right end of the ship
            right_end = int(np.max(curr_ship[:, 1]))
            # print('right_end is : ',right_end)
            if right_end < upper_limit:
                forbidden.append([x, right_end + 1])
        if orient == "ver":
            # Set y as constant
            y = int(curr_ship[0][1])
            # Find minimum row number as the upper end of the ship
            upper_end = int(np.min(curr_ship[:, 0]))
            # print('upper_end is : ', upper_end)
            if upper_end > 0:
                forbidden.append([upper_end - 1, y])
            # Find maximum row number as the lower end of the ship
            lower_end = int(np.max(curr_ship[:, 0]))
            # print('lower_end is : ', lower_end)
            if lower_end < upper_limit:
                forbidden.append([lower_end + 1, y])
        return forbidden

    seg_2_fields = []
    seg_3_fields = []
    seg_4_fields = []
    seg_5_fields = []
    forbidden = []
    all_ships = []
    #temp_ship = []
    # print ('working with board:')
    # print (board)
    seg_2_fields, seg_3_fields, seg_4_fields, seg_5_fields = count_fields (board)
    #print (seg_2_fields, seg_3_fields, seg_4_fields, seg_5_fields)

    if len(seg_2_fields) == n_seg2*2 and len(seg_3_fields) == n_seg3*3 and  len(seg_4_fields) == n_seg4*4 and len(seg_5_fields) == n_seg5*5:
        # print ('seg3 fields: ',seg_3_fields)
        seg2g = group_2segs(seg_2_fields)
        seg3g = group_3segs(seg_3_fields)
        seg4g = group_4segs(seg_4_fields)
        seg5g = group_5segs(seg_5_fields)

        gr = [seg2g,seg3g,seg4g,seg5g]
        if False in gr:
            all_ships_grouped = False
        else:
            all_ships_grouped = True

        #all_ships_grouped = group_3segs (seg_3_fields)
        # print('all_ships_grouped: ', all_ships_grouped)
        if all_ships_grouped == True:
            # print('all ships: ', all_ships)
            for k in all_ships:
                curr_ship = np.array(k)
                # print(curr_ship)

                orient = identify_orientation (curr_ship)
                # print('orient is: ', orient)
                # print(upper_limit)
                forbid_endings (board, curr_ship, orient, upper_limit)
                forbid_diagonals (board, curr_ship)

            # print('forbidden fields: ', forbidden)

            board_with_forbidden = copy.deepcopy(board)

            validated = True
            for field in forbidden:
                x=field[0]
                y=field[1]
                board_with_forbidden[x][y] = 9
                if board [x][y] > 1 and board [x][y] < 6:
                    validated = False
                    print(board)
                    print('Ships are touching or overlapping!')
                    break
            if validated == True:
                dum = 1
                #print(board_with_forbidden)
        else:
            validated = False
            print('Ships could not be identified correctly')
    else:
        validated = False
        print('Incorrect number of ship segments')

    print('Validated: ', validated)
    return validated






#validate(board)