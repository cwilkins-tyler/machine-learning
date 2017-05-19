import time
import pygame
import random

WINDOW_SIZE = WIDTH, HEIGHT = 640, 700
MARGIN_COLOR = 128, 102, 69
GSIZE = WIDTH // 12
MARGIN = GSIZE // 12
SPECIALSQS = {(5, 5), (0, 0), (0, 10), (10, 10), (10, 0)}
WINNERSQS = SPECIALSQS.difference([(5, 5)])
INITIAL_STATE = [[4, 0, 0, 1, 1, 1, 1, 1, 0, 0, 4],
                 [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                 [1, 0, 0, 0, 0, 2, 0, 0, 0, 0, 1],
                 [1, 0, 0, 0, 2, 2, 2, 0, 0, 0, 1],
                 [1, 1, 0, 2, 2, 3, 2, 2, 0, 1, 1],
                 [1, 0, 0, 0, 2, 2, 2, 0, 0, 0, 1],
                 [1, 0, 0, 0, 0, 2, 0, 0, 0, 0, 1],
                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                 [4, 0, 0, 1, 1, 1, 1, 1, 0, 0, 4]]


class Board(object):

    """The Board class contains information about the physical board.

    Whereas the move class contains information about the state of the pieces,
    the Board class contains information for the look of the board.
    """

    def __init__(self, state):
        """Create a playing board and color code it.

        Attributes:
            grid (list(str)): A list of strings which classify each tile. Each
                              char in the string maps to a type of tile:
                                  4 -> corner tile
                                  1 -> initial attack tile
                                  2 -> initial defend tile
                                  3 -> center tile
            colors (dict): Maps the tiles to the appropriate color.
            dim (int): Dimension of the board (i.e. num of rows or cols.)
            piece (int): Size of playing piece.
        """
        self.grid = state

        self.colors = {4: (25, 25, 25),
                       1: (186, 169, 85),
                       2: (218, 185, 23),
                       3: (242, 240, 228),
                       0: (250, 236, 163)}

        self.dim = len(self.grid)
        self.agent = None

    def initialize_pieces(self):
        """Create all of the game pieces and put them in groups.

        Note:
            The board layout from Board class is used for initial placement of
            pieces.

        Args:
            board (Board): the game board object
        """
        for y in range(self.dim):
            for x in range(self.dim):
                p = self.grid[y][x]
                if p == 1:
                    Attacker(x, y)
                elif p == 2:
                    Defender(x, y)
                elif p == 3:
                    King(x, y)

    @staticmethod
    def cleanup():
        """Empty out all groups of sprites"""
        Current.empty()
        Kings.empty()
        Defenders.empty()
        Attackers.empty()
        Pieces.empty()

    def get_current_state(self):
        board_state = []
        for y in range(self.dim):
            board_state.append([])
            for x in range(self.dim):
                piece_found = False
                piece_type = 0
                for piece in Attackers:
                    if piece.x_tile == x:
                        if piece.y_tile == y:
                            piece_found = True
                            piece_type = 1

                for piece in Defenders:
                    if piece.x_tile == x:
                        if piece.y_tile == y:
                            piece_found = True
                            piece_type = 2

                if (x, y) in SPECIALSQS:
                    piece_type = 4

                for piece in Kings:
                    if piece.x_tile == x:
                        if piece.y_tile == y:
                            piece_found = True
                            piece_type = 3

                if piece_found:
                    board_state[y].append(piece_type)
                else:
                    board_state[y].append(piece_type)

        return board_state

    def dump_board(self):
        current_state = self.get_current_state()
        print('\n'.join(' '.join(map(str, sl)) for sl in current_state))

    @staticmethod
    def get_all_valid_actions():
        # make a board from the current state
        all_actions = []
        for piece in Attackers:
            valid_moves = piece.get_valid_moves()
            for valid_move in valid_moves:
                move_representation = (piece.x_tile, piece.y_tile, valid_move[0], valid_move[1])
                all_actions.append(move_representation)

        return sorted(all_actions)


class Move(object):

    """The Move class contains all information about the current move state."""

    def __init__(self):
        """Initialized the Move object.

        a_turn: Bool which is true when its the Attacker's turn, false o.w.
        selected: Bool which is true if a piece has been selected to move.
        king_killed: Bool which is true if the king has been killed.
        escaped: Bool which is true if the king escaped
        game_over: Bool which is true if either player has won or its a draw
        restart: Bool which pauses game and asks if players want to restart
        """
        self.a_turn = True
        self.selected = False
        self.king_killed = False
        self.escaped = False
        self.game_over = False
        self.restart = False

    def agent_move(self, board, agent_choice):
        # move is in the format (chosen_piece.x, chosen_piece.y, chosen_move.x, chosen_move.y)
        chosen_piece, valid_moves = self.select_specific_piece(Attackers, (agent_choice[0], agent_choice[1]))

        if chosen_piece:
            self.select(chosen_piece)

            chosen_move = (agent_choice[2], agent_choice[3])
            chosen_pos = self.ppos_cent(chosen_move[0], chosen_move[1])
            Current.add(chosen_piece)

            if self.is_valid_move(chosen_pos, Current.sprites()[0]):
                self.finalise_move(board)
            else:
                print('Final check failed')
        else:
            print('Invalid agent piece: ({}, {})'.format(chosen_piece.x_tile, chosen_piece.y_tile))
            board.dump_board()
            raise Exception('Invalid piece')

    def computer_move(self, board):
        if self.game_over:
            pass
        elif self.restart:
            pass
        else:
            if self.a_turn:
                piece, valid_moves, best_path = self.select_computer_piece(Attackers)
                chosen_pos = self.select_computer_move(valid_moves, piece, best_path)
                if self.is_valid_move(chosen_pos, Current.sprites()[0]):
                    self.finalise_move(board)
            else:
                piece, valid_moves, best_path = self.select_computer_piece(Defenders)
                chosen_pos = self.select_computer_move(valid_moves, piece, best_path)

                if self.is_valid_move(chosen_pos, Current.sprites()[0]):
                    self.finalise_move(board)
                else:
                    print('Not a valid move')

    @staticmethod
    def select_specific_piece(sprite_group, target_coords):
        piece = None
        valid_moves = []
        for sprite in sprite_group.sprites():
            if sprite.x_tile == target_coords[0] and sprite.y_tile == target_coords[1]:
                piece = sprite
                valid_moves = piece.get_valid_moves()
                break
        return piece, valid_moves

    def select_computer_piece(self, sprite_group):
        best_path = []
        if self.a_turn:
            # random moves for the attackers for now
            piece, valid_moves = self.select_random_piece(sprite_group)
        else:
            # check if we need to free the king
            if self.king_trapped():
                # move pieces out of the way so the king can move, this will usually only happen in the first few moves
                piece, valid_moves = self.move_trapping_pieces()
            else:
                # find a path from the king to a special piece
                king = Kings.sprites()[0]
                king_cell = Cell((king.x_tile, king.y_tile))
                all_paths = []
                for end_square in WINNERSQS:
                    path = self.find_path(king_cell, end_square)

                    if len(path) > 0:
                        all_paths.append(path)

                if len(all_paths) > 0:
                    # move the king
                    piece = king
                    valid_moves = piece.get_valid_moves()
                    best_path = sorted(all_paths, key=lambda x: len(x))[0]
                else:
                    # try to take a piece from the other side
                    piece, valid_moves = self.find_captures(sprite_group)
                    if not piece:
                        piece, valid_moves = self.select_random_piece(sprite_group)

        self.select(piece)
        return piece, valid_moves, best_path

    @staticmethod
    def select_random_piece(sprite_group):
        piece = None
        valid_moves = []
        valid_move_count = 0
        while not valid_move_count:
            piece = random.choice(sprite_group.sprites())
            valid_moves = piece.get_valid_moves()
            valid_move_count = len(valid_moves)
        return piece, valid_moves

    def select_computer_move(self, valid_moves, piece, best_path):
        if len(valid_moves.intersection(WINNERSQS)) > 0:
            # we can move onto the special squares, so do that
            chosen_move = random.choice(tuple(valid_moves.intersection(WINNERSQS)))
        else:
            if len(best_path) > 0:
                for path in best_path:
                    if path.coords in valid_moves:
                        chosen_move = path.coords
            else:
                chosen_move = random.choice(tuple(valid_moves))
        chosen_pos = self.ppos_cent(chosen_move[0], chosen_move[1])
        Current.add(piece)
        return chosen_pos

    @staticmethod
    def piece_trapped(piece):
        valid_moves = piece.get_valid_moves()
        valid_move_count = len(valid_moves)
        if valid_move_count == 0:
            return True
        else:
            return False

    def king_trapped(self):
        king_piece = Kings.sprites()[0]
        return self.piece_trapped(king_piece)

    @staticmethod
    def heuristic(a, b):
        # Manhattan distance on a square grid
        return abs(a.x - b.x) + abs(a.y - b.y)

    @staticmethod
    def next_step(open_list):
        open_list.sort(key=lambda x: x.f, reverse=True)
        cell = open_list.pop()
        cell.get_neighbours()
        return cell

    @staticmethod
    def already_seen_cell(cell, cell_list):
        already_seen = False
        for open_cell in cell_list:
            if open_cell.coords == cell.coords:
                already_seen = True

        return already_seen

    def find_path(self, start, goal):
        target = Cell(goal)

        open_list = []
        closed_list = []
        open_list.append(start)
        iteration = 0
        while len(open_list) > 0:
            current = self.next_step(open_list)
            if current.coords == target.coords:
                cell = current
                path = []
                while cell.parent:
                    path.append(cell)
                    cell = cell.parent

                path.reverse()
                return path

            closed_list.append(current)

            for cell in current.neighbours:
                if self.already_seen_cell(cell, closed_list):
                    continue

                already_seen = self.already_seen_cell(cell, open_list)

                # check for blocks
                if not already_seen:
                    cell.distance_from_target = self.heuristic(cell, target)
                    cell.parent = current
                    cell.g = cell.parent.g + 1
                    cell.f = cell.g + cell.distance_from_target
                    open_list.append(cell)

            iteration += 1
        else:
            return []

    @staticmethod
    def can_complete_capture(target_move, target_capture):
        target_cell = ''
        # Up
        if target_move.y + 1 == target_capture.y:
            target_cell = Cell((target_move.x, target_move.y + 2))

        # Right
        if target_move.x + 1 == target_capture.x:
            target_cell = Cell((target_move.x + 2, target_move.y))

        # Down
        if target_move.y - 1 == target_capture.y:
            target_cell = Cell((target_move.x, target_move.y - 2))

        # Left
        if target_move.x - 1 == target_capture.x:
            target_cell = Cell((target_move.x - 2, target_move.y))

        if target_cell:
            if target_cell.cell_has_defending_piece((target_cell.x, target_cell.y)):
                return True
            elif target_cell.cell_is_special((target_cell.x, target_cell.y)):
                return True

    def find_captures(self, sprite_group):
        capturing_pieces = dict()
        for piece in sprite_group:
            valid_moves = piece.get_valid_moves()
            for valid_move in valid_moves:
                target_cell = Cell(valid_move)
                target_pieces = target_cell.get_neighbouring_pieces()
                if len(target_pieces) > 0:
                    for target_piece in target_pieces:
                        # check if the piece is from the opposition
                        if target_piece.cell_has_attacking_piece((target_piece.x, target_piece.y)):
                            if self.can_complete_capture(target_cell, target_piece):
                                capturing_pieces.setdefault(piece, [])
                                capturing_pieces[piece].append(valid_move)

        if len(list(capturing_pieces.keys())) > 0:
            chosen_piece = random.choice(list(capturing_pieces.keys()))
            chosen_moves = set(capturing_pieces[chosen_piece])
        else:
            chosen_piece = 0
            chosen_moves = 0

        return chosen_piece, chosen_moves

    def finalise_move(self, board):
        if Current.sprites()[0] in Kings:
            self.king_escaped(Kings)
        if self.a_turn:
            self.remove_pieces(Defenders, Attackers, Kings)
        else:
            self.remove_pieces(Attackers, Defenders, Kings)

        self.end_turn(Current.sprites()[0])
        Current.empty()
        # dump_board(board)

    def find_valid_neighbours(self, piece):
        valid_neighbours = []
        neighbours = piece.find_neighbours()
        for np in neighbours:
            if not self.piece_trapped(np):
                if piece in Defenders:
                    valid_neighbours.append(np)
        return valid_neighbours

    def move_trapping_pieces(self):
        for king_piece in Kings:
            neighbours = king_piece.find_neighbours()

            # check if all the neighbours are trapped too
            valid_neighbours = []
            for np in neighbours:
                if not self.piece_trapped(np):
                    if np in Defenders:
                        valid_neighbours.append(np)

            if len(valid_neighbours) == 0:
                for neighbour_piece in neighbours:
                    valid_next_neighbours = self.find_valid_neighbours(neighbour_piece)
                    if len(valid_next_neighbours) > 0:
                        next_neighbour_piece = random.choice(valid_next_neighbours)
                        valid_moves = next_neighbour_piece.get_valid_moves()
                        piece = next_neighbour_piece
                        return piece, valid_moves
            else:
                neighbour_piece = random.choice(valid_neighbours)
                valid_moves = neighbour_piece.get_valid_moves()
                piece = neighbour_piece
                return piece, valid_moves

    def select(self, piece):
        """Allow players to select one of their pieces to move.

        When a player clicks on a piece, this function first checks if they
        have chosen a piece to move already. If they have not, the function
        determines if the piece is theirs or not, and if it is, its location
        is stored in the Move object, its color is changed so the player can
        see what they selected, and the valid moves for that piece are
        calculated.

        Args:
            piece (Piece): the playing piece that the player selected.
        """
        if not self.selected:
            self.selected = True
            piece.color = (71, 166, 169)
            self.row = piece.x_tile
            self.col = piece.y_tile
            self.vm = self.valid_moves(piece.special_sqs)
        else:
            self.selected = False
            piece.color = piece.base_color

    def valid_moves(self, special_sqs):
        """Determine the valid moves for the selected piece.

        Currently there are four very similar functions to determine all valid
        moves to the left, the right, above, and below. These functions need to
        be refactored.

        Args:
            special_sqs (bool): True if piece can move on special squares

        Returns:
            vm (set(int,int)): Set of valid moves.
        """
        vm = set()
        vm.update(self.left_bound())
        vm.update(self.right_bound())
        vm.update(self.up_bound())
        vm.update(self.down_bound())
        if not special_sqs:
            vm.difference_update(SPECIALSQS)
        return vm

    def is_valid_move(self, pos, piece):
        """Determine if the selected move is valid or not.

        The function finds the tile on the board where the player wants to move
        their piece and returns True if the tile is in the set of valid moves
        for the selected piece.

        Args:
            pos (int, int): x and y coordinates in pixels
            piece (Piece): the selected piece

        Returns:
            bool: True if valid move, false o.w.
        """
        row = pos[0] // (WIDTH // 11)
        col = pos[1] // (WIDTH // 11)
        if (row, col) in self.vm:
            piece.pos_cent(row, col)
            self.row = row
            self.col = col
            return True
        else:
            print('Invalid move')
            return False

    def king_escaped(self, Kings):
        """Check if king has moved onto a corner square."""
        king = (Kings.sprites()[0].x_tile, Kings.sprites()[0].y_tile)
        if king in WINNERSQS:
            self.escaped = True
            self.game_over = True

    def remove_pieces(self, g1, g2, Kings):
        """Determine if any pieces need to be removed from the board.

        check_pts is a list of four tuples. Each tuple is a tuple of tile
        coordinates, the first of which is directly next to the square
        where the piece moved, and the second of which is two squares away
        in the same direction. First, the function checks to see if there is
        an opponent's piece adjacent to where the player just moved his
        piece. If there is one, then it checks if the other side of the
        opponent's piece is either occupied by the player's piece or if it
        is an unoccupied hostile territory (SPECIALSQS). If either of those
        is true, then the piece is captured, and removed from the board.

        Args:
            g1 (Group(sprites)): the opponent's pieces
            g2 (Group(sprites)): the current player's pieces
            Kings (Group(sprites)): the group containing the king
        """
        check_pts = set([((self.row, self.col + 1), (self.row, self.col + 2)),
                         ((self.row + 1, self.col), (self.row + 2, self.col)),
                         ((self.row, self.col - 1), (self.row, self.col - 2)),
                         ((self.row - 1, self.col), (self.row - 2, self.col))])
        captured = []
        king = (Kings.sprites()[0].x_tile, Kings.sprites()[0].y_tile)
        for square in check_pts:
            if square[0] == king:
                if Kings.sprites()[0] in g1:
                    if self.kill_king(king[0], king[1], g2):
                        self.king_killed = True
                        self.game_over = True
                        captured.append(Kings.sprites()[0])
            else:
                for p1 in g1:
                    if (p1.x_tile, p1.y_tile) == square[0]:
                        for p2 in g2:
                            if (p2.x_tile, p2.y_tile) == (square[1]):
                                captured.append(p1)
                            elif square[1] in SPECIALSQS:
                                if square[1] != king:
                                    captured.append(p1)
        for a in captured:
            a.kill()

    @staticmethod
    def kill_king(x, y, attackers):
        """Determine if the king has been killed.

        The king is killed if it is surrounded on all four sides by attacking
        pieces or hostile territories.

        Args:
            x (int): x tile coordinate of the king
            y (int): y tile coordinate of the king

        Returns:
            True if king has been killed, False o.w.
        """
        kill_pts = set([(x, y + 1), (x + 1, y), (x, y - 1), (x - 1, y)])
        kill_pts.difference_update(SPECIALSQS)
        attack_pts = set()
        for pt in kill_pts:
            for a in attackers:
                try:
                    attack_pts.add((a.x_tile, a.y_tile))
                except KeyError:
                    pass
        kill_pts.difference_update(attack_pts)
        if not kill_pts:
            return True

    def left_bound(self):
        """Find the all valid moves to the left of the selected piece.

        Iterates through all spaces to the left until the space is either
        occupied or it has reached the edge of the board. Every valid move
        is added to a set of valid moves, which is returned.

        Returns:
            vm (set(int, int)): Set of tuples of the tile coordinates of
                                valid moves to the left.
        """
        vm = set()
        temp_row = self.row - 1
        while True:
            if temp_row < 0:
                return vm
            for p in Pieces:
                ppos = self.ppos_cent(temp_row, self.col)
                if p.rect.collidepoint(ppos):
                    return vm
            vm.add((temp_row, self.col))
            temp_row -= 1

    def right_bound(self):
        """Find the all valid moves to the right of the selected piece.

        Iterates through all spaces to the right until the space is either
        occupied or it has reached the edge of the board. Every valid move
        is added to a set of valid moves, which is returned.

        Returns:
            vm (set(int, int)): Set of tuples of the tile coordinates of
                                valid moves to the right.
        """
        vm = set()
        temp_row = self.row + 1
        clear = True
        while clear:
            if temp_row > 10:
                return vm
            for p in Pieces:
                ppos = self.ppos_cent(temp_row, self.col)
                if p.rect.collidepoint(ppos):
                    return vm
            vm.add((temp_row, self.col))
            temp_row += 1

    def up_bound(self):
        """Find the all valid moves above the selected piece.

        Iterates through all spaces above until the space is either
        occupied or it has reached the edge of the board. Every valid move
        is added to a set of valid moves, which is returned.

        Returns:
            vm (set(int, int)): Set of tuples of the tile coordinates of
                                valid moves above.
        """
        vm = set()
        temp_col = self.col - 1
        clear = True
        while clear:
            if temp_col < 0:
                return vm
            for p in Pieces:
                ppos = self.ppos_cent(self.row, temp_col)
                if p.rect.collidepoint(ppos):
                    return vm
            vm.add((self.row, temp_col))
            temp_col -= 1

    def down_bound(self):
        """Find the all valid moves below the selected piece.

        Iterates through all spaces below until the space is either
        occupied or it has reached the edge of the board. Every valid move
        is added to a set of valid moves, which is returned.

        Returns:
            vm (set(int, int)): Set of tuples of the tile coordinates of
                                valid moves below.
        """
        vm = set()
        temp_col = self.col + 1
        clear = True
        while clear:
            if temp_col > 10:
                return vm
            for p in Pieces:
                ppos = self.ppos_cent(self.row, temp_col)
                if p.rect.collidepoint(ppos):
                    return vm
            vm.add((self.row, temp_col))
            temp_col += 1

    @staticmethod
    def ppos(x):
        """Find the top or left pixel position of a given tile.

        Args:
            x (int): the row or column number

        Returns:
            (int): the top or left pixel location of the tile.
        """
        return x*(GSIZE + (GSIZE // 12)) + (GSIZE // 12)

    def ppos_cent(self, x, y):
        """Find the center pixel position of a given tile.

        Args:
            x (int): the row number
            y (int): the column number

        Returns:
            (int, int): tuple of the center pixel location of tile
        """
        return self.ppos(x) + (GSIZE // 2), self.ppos(y) + (GSIZE // 2)

    def end_turn(self, piece):
        """Perform some cleanup to end the turn.

        Once the turn has been completed, the a_turn bool is flipped so the
        other player can go, the selected piece is deselected, and its color
        returns to normal.
        """
        self.a_turn = not self.a_turn
        self.selected = False
        piece.color = piece.base_color


def initialize_groups():
    """Create global groups for different pieces.

    Notes:
        The groups are defined as follows:
            Pieces: all pieces
            Attackers: all attacking pieces
            Defenders: all defending pieces, including king
            Kings: the king piece
            Current: the current selected piece
    """
    global Pieces
    global Attackers
    global Defenders
    global Kings
    global Current

    Pieces = pygame.sprite.Group()
    Attackers = pygame.sprite.Group()
    Defenders = pygame.sprite.Group()
    Kings = pygame.sprite.Group()
    Current = pygame.sprite.Group()

    Piece.groups = Pieces
    Defender.groups = Pieces, Defenders
    Attacker.groups = Pieces, Attackers
    King.groups = Pieces, Defenders, Kings


class Piece(pygame.sprite.Sprite):

    """Class for all playing pieces.

    Pieces are pygame sprite objects. It makes grouping, determining
    collisions, and removing pieces very simple.
    """

    def __init__(self, x, y):
        """Create a piece at a given location.

        special_sqs, a bool which determines if a piece is allowed to move
        onto the corners or center square, defaults to False.

        Args:
            x (int): the row that the piece will be placed
            y (int): the column that the piece will be placed.
        """
        pygame.sprite.Sprite.__init__(self, self.groups)
        self.special_sqs = False
        self.pos_cent(x, y)

    def get_valid_moves(self):
        temp_move = Move()
        temp_move.select(self)
        valid_moves = temp_move.valid_moves(self.special_sqs)
        temp_move.select(self)
        return valid_moves

    @staticmethod
    def pos(x):
        """Find the top or left pixel position of a given tile.

        Args:
            x (int): the row or column number

        Returns:
            (int): the top or left pixel location of the tile.
        """
        return x*(GSIZE + (GSIZE // 12)) + (GSIZE // 12)

    def pos_cent(self, x, y):
        """Find the center pixel position of a given tile.

        Stores coordinates of the tile (x_tile and y_tile) as well as the
        center pixel locations of the tiles.

        Args:
            x (int): the row number
            y (int): the column number

        Returns:
            (int, int): tuple of the center pixel location of tile
        """
        self.x_tile = x
        self.y_tile = y
        self.x_px = self.pos(x) + (GSIZE // 2)
        self.y_px = self.pos(y) + (GSIZE // 2)
        self.rect = pygame.Rect([self.x_px - GSIZE/2,
                                 self.y_px - GSIZE/2,
                                 GSIZE,
                                 GSIZE])

    def draw(self, screen):
        """Draw the piece on the board in the correct location."""
        pygame.draw.circle(screen, self.color, [self.x_px, self.y_px], int(GSIZE/2))

    def find_neighbours(self):
        neighbours = []
        for neighbour_piece in Pieces:
            if neighbour_piece.x_tile == self.x_tile + 1 and neighbour_piece.y_tile == self.y_tile:
                neighbours.append(neighbour_piece)
            elif neighbour_piece.x_tile == self.x_tile - 1 and neighbour_piece.y_tile == self.y_tile:
                neighbours.append(neighbour_piece)
            elif neighbour_piece.x_tile == self.x_tile and neighbour_piece.y_tile == self.y_tile + 1:
                neighbours.append(neighbour_piece)
            elif neighbour_piece.x_tile == self.x_tile and neighbour_piece.y_tile == self.y_tile - 1:
                neighbours.append(neighbour_piece)
        return neighbours


class Attacker(Piece):

    """Class for all attacking pieces; inherits from Piece."""

    def __init__(self, x, y):
        """Inherit from Piece and give attacking piece a color."""
        Piece.__init__(self, x, y)
        pygame.sprite.Sprite.__init__(self, self.groups)
        self.base_color = (149, 19, 62)
        self.color = self.base_color


class Defender(Piece):

    """Class for all defending pieces; inherits from Piece."""

    def __init__(self, x, y):
        """Inherit from Piece and give attacking piece a color."""
        Piece.__init__(self, x, y)
        pygame.sprite.Sprite.__init__(self, self.groups)
        self.base_color = (52, 134, 175)
        self.color = self.base_color


class King(Defender):

    """Class for king; inherits from Defender."""

    def __init__(self, x, y):
        """Inherit from Piece and give attacking piece a color.

        The king can move on special squares, so special_sqs is True.
        """
        Defender.__init__(self, x, y)
        pygame.sprite.Sprite.__init__(self, self.groups)
        self.base_color = (19, 149, 62)
        self.color = self.base_color
        self.special_sqs = True


class Cell():
    def __init__(self, coords):
        self.x, self.y = coords
        self.coords = coords
        self.neighbours = []
        self.parent = 0
        self.wall = 0
        self.f = 0
        self.g = 0
        self.distance_from_target = 0

    def cell_has_piece(self, coords):
        if self.cell_has_attacking_piece(coords):
            return True

        if self.cell_has_defending_piece(coords):
            return True
        return False

    @staticmethod
    def cell_has_attacking_piece(coords):
        for piece in Attackers:
            if piece.x_tile == coords[0] and piece.y_tile == coords[1]:
                return True

    @staticmethod
    def cell_has_defending_piece(coords):
        for piece in Defenders:
            if piece.x_tile == coords[0] and piece.y_tile == coords[1]:
                return True

        return False

    @staticmethod
    def cell_is_special(coords):
        if coords in SPECIALSQS:
            return True

        return False

    def get_neighbours(self):
        self.neighbours = []
        # Up
        if self.y < 10 and (not self.cell_has_piece((self.x, self.y + 1))):
            self.neighbours.append(Cell((self.x, self.y + 1)))
        # Right
        if self.x < 10 and (not self.cell_has_piece((self.x + 1, self.y))):
            self.neighbours.append(Cell((self.x + 1, self.y)))
        # Down
        if self.y > 0 and (not self.cell_has_piece((self.x, self.y - 1))):
            self.neighbours.append(Cell((self.x, self.y - 1)))
        # Left
        if self.x > 0 and (not self.cell_has_piece((self.x - 1, self.y))):
            self.neighbours.append(Cell((self.x - 1, self.y)))

    def get_neighbouring_pieces(self):
        neighbour_pieces = []
        # Up
        if self.y < 10 and self.cell_has_piece((self.x, self.y + 1)):
            neighbour_pieces.append(Cell((self.x, self.y + 1)))
        # Right
        if self.x < 10 and self.cell_has_piece((self.x + 1, self.y)):
            neighbour_pieces.append(Cell((self.x + 1, self.y)))
        # Down
        if self.y > 0 and self.cell_has_piece((self.x, self.y - 1)):
            neighbour_pieces.append(Cell((self.x, self.y - 1)))
        # Left
        if self.x > 0 and self.cell_has_piece((self.x - 1, self.y)):
            neighbour_pieces.append(Cell((self.x - 1, self.y)))

        return neighbour_pieces


class Simulator():
    def __init__(self, sim_mode=True, visualise=False):
        self.sim_mode = sim_mode
        self.visualise = visualise
        pygame.init()
        pygame.display.set_caption("King's Table")
        if visualise:
            self.screen = pygame.display.set_mode(WINDOW_SIZE)

        initialize_groups()
        self.board = Board(INITIAL_STATE)
        self.move = Move()
        self.board.initialize_pieces()
        self.round_number = 0

    @staticmethod
    def update_image(screen, board, move, text, text2, sim_mode):
        """Update the image that the users see.

        Note:
            Right now, it redraws the whole board every time it goes through this
            function. In the future, it should only update the necessary tiles.

        Args:
            screen (pygame.Surface): game window that the user interacts with
            board (Board): the board that the pieces are on
            move (Move): the move state data
        """
        screen.fill(MARGIN_COLOR)
        for y in range(board.dim):
            for x in range(board.dim):
                xywh = [x*(GSIZE + MARGIN) + MARGIN,
                        y*(GSIZE + MARGIN) + MARGIN,
                        GSIZE,
                        GSIZE]
                pygame.draw.rect(screen, board.colors[board.grid[x][y]], xywh)

        for piece in Pieces:
            piece.draw(screen)

        if not sim_mode:
            """Write which player's turn it is on the bottom of the window."""
            font = pygame.font.Font(None, 36)
            msg = font.render(text, 1, (0, 0, 0))
            msgpos = msg.get_rect()
            if text2:
                msg2 = font.render(text2, 1, (0, 0, 0))
                msgpos2 = msg2.get_rect()
                msgpos.centerx = screen.get_rect().centerx
                msgpos.centery = ((HEIGHT - WIDTH) / 7) + WIDTH
                msgpos2.centerx = screen.get_rect().centerx
                msgpos2.centery = (5 * (HEIGHT - WIDTH) / 7) + WIDTH
                screen.blit(msg, msgpos)
                screen.blit(msg2, msgpos2)
            else:
                msgpos.centerx = screen.get_rect().centerx
                msgpos.centery = ((HEIGHT - WIDTH) / 2) + WIDTH
                screen.blit(msg, msgpos)

        pygame.display.flip()
        pygame.event.pump()

    def step(self, action):
        self.round_number += 1
        self.move.agent_move(self.board, action)

        if self.visualise:
            self.update_image(self.screen, self.board, self.move, '', '', self.sim_mode)
            # slow down the updates if the screen is being shown
            time.sleep(1)

        self.move.computer_move(self.board)

        current_state = self.board.get_current_state()
        reward = float(self.round_number) / 200
        if self.move.game_over:
            if self.move.king_killed:
                print("winner=Attacker")
                reward = 1
            else:
                print("winner=Defender")
                reward = -1

        return self.move, current_state, reward

    def get_state(self):
        return self.board.get_current_state()

    def get_all_valid_actions(self):
        all_moves = self.board.get_all_valid_actions()
        return all_moves

    def game_over(self):
        self.board.cleanup()
        pygame.display.quit()
        pygame.quit()
