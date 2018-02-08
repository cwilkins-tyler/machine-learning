"""
A two player version of Hnefatafl, a Viking board game.
A full description of the game can be found here: http://tinyurl.com/2lpvjb
Author: Sean Lowen
Date: 7/13/2015
"""

import sys
import pygame
from pygame.locals import *
from kt_simulator import *


def run_game(screen, player_is_attacker, player_is_defender, sim_mode=False, sim_env=None):
    """Start and run a new game of hnefatafl.
    The game, groups, board, move info, screen, and pieces are initialized
    first. Then, the game starts. It runs in a while loop, which will exit
    if the user closes out of the game. Another event that it listens
    for is a MOUSEBUTTONDOWN event; the game takes action when the user clicks
    on the board. If a piece has not been selected yet and the user clicks on
    one of his pieces, then the piece will be selected and change colors. The
    user can click on this piece again to deselect it, or they can click
    on a square that is a valid move for that piece. If it is a valid move,
    the piece will move there and it is the next person's turn. The game
    also listens for KEYDOWN event. If the game has ended or the player wants
    to restart the game, it will listen for 'y' or 'n'. If the player wants
    to restart the game, they can press 'r', which will require confirmation
    before actually restarting.
    Args:
        screen (pygame.Surface): The game window
    Returns:
        True if players want a new game, False o.w.
    """
    sim = Simulator(visualise=False)

    while 1:
        """Text to display on bottom of game."""
        text2 = None
        if sim.move.a_turn:
            text = "Attacker's Turn"
        if not sim.move.a_turn:
            text = "Defender's Turn"
        if sim.move.escaped:
            text = "King escaped! Defenders win!"
            text2 = "Play again? y/n"
        if sim.move.king_killed:
            text = "King killed! Attackers win!"
            text2 = "Play again? y/n"
        if sim.move.restart:
            text = "Restart game? y/n"
        if sim.move.escaped:
            text = ""

        if not player_is_attacker and not player_is_defender:
            # we are in self play mode
            if sim_mode:
                if sim.move.game_over:
                    current_state = sim.board.get_current_state()
                    if sim.move.king_killed:
                        sim.board.agent.update(current_state, sim.board.round, winner='Attacker')
                    else:
                        sim.board.agent.update(current_state, sim.board.round, winner='Defender')
                    return True
                sim.board.agent = sim_env.primary_agent
                sim.round_number += 1
            if sim.move.a_turn and sim_mode:
                current_state = sim.board.get_current_state()
                all_moves = sim.board.get_all_valid_actions(sim.move.a_turn)
                chosen_action = sim.board.agent.update(current_state, sim.board.round, valid_moves=all_moves)
                agent_move(sim.move, sim.board, chosen_action)
            else:
                sim.move.computer_move(sim.board)

        for event in pygame.event.get():
            if event.type == QUIT:
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if sim.move.game_over and event.key == pygame.K_n:
                    return False
                if sim.move.game_over and event.key == pygame.K_y:
                    return True
                if sim.move.restart and event.key == pygame.K_n:
                    sim.move.restart = False
                if sim.move.restart and event.key == pygame.K_y:
                    return True
                if event.key == pygame.K_r:
                    sim.move.restart = True
            elif sim_mode:
                if sim.move.game_over:
                    return True
            else:
                if sim.move.a_turn:
                    sim.round_number += 1
                    if player_is_attacker:
                        sim.move.player_move(event, sim.board)
                    else:
                        sim.move.computer_move(sim.board)
                else:
                    sim.round_number += 1
                    if player_is_defender:
                        sim.move.player_move(event, sim.board)
                    else:
                        sim.move.computer_move(sim.board)

        sim.update_image(screen, sim.board, sim.move, text, text2, sim_mode)


class MenuItem(pygame.font.Font):
    def __init__(self, text, font=None, font_size=30,
                 font_colour=(255, 255, 255), pos_x=0, pos_y=0):
        pygame.font.Font.__init__(self, font, font_size)
        self.text = text
        self.font_size = font_size
        self.font_colour = font_colour
        self.label = self.render(self.text, 1, self.font_colour)
        self.width = self.label.get_rect().width
        self.height = self.label.get_rect().height
        self.pos_x = pos_x
        self.pos_y = pos_y
        self.position = pos_x, pos_y

    def set_position(self, x, y):
        self.position = (x, y)
        self.pos_x = x
        self.pos_y = y

    def is_mouse_selection(self, pos_x, pos_y):
        if (self.pos_x <= pos_x <= self.pos_x + self.width) and (self.pos_y <= pos_y <= self.pos_y + self.height):
            return True
        return False

    def set_font_colour(self, rgb_tuple):
        self.font_colour = rgb_tuple
        self.label = self.render(self.text, 1, self.font_colour)


WHITE = (255, 255, 255)
RED = (255, 0, 0)
BLACK = (0, 0, 0)


class GameMenu:
    def __init__(self, screen, items, bg_colour=BLACK, font=None, font_size=30, font_colour=WHITE):
        self.screen = screen
        self.scr_width = self.screen.get_rect().width
        self.scr_height = self.screen.get_rect().height
        self.bg_colour = bg_colour
        self.clock = pygame.time.Clock()

        self.font = pygame.font.SysFont(font, font_size)
        self.font_colour = font_colour
        self.selection = ''
        self.items = []
        for index, item in enumerate(items):
            menu_item = MenuItem(item)

            total_height = len(items) * menu_item.height
            pos_x = (self.scr_width / 2) - (menu_item.width / 2)
            pos_y = (self.scr_height / 2) - (total_height / 2) + ((index * 2) + index * menu_item.height)

            menu_item.set_position(pos_x, pos_y)
            self.items.append(menu_item)

    def run(self):
        mainloop = True
        while mainloop:
            # Limit frame speed to 50fps
            self.clock.tick(50)
            mouse_pos = pygame.mouse.get_pos()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    mainloop = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    for item in self.items:
                        if item.is_mouse_selection(mouse_pos[0], mouse_pos[1]):
                            self.selection = item.text
                            mainloop = False

            self.screen.fill(self.bg_colour)
            for item in self.items:
                if item.is_mouse_selection(mouse_pos[0], mouse_pos[1]):
                    item.set_font_colour(RED)
                else:
                    item.set_font_colour(WHITE)
                self.screen.blit(item.label, item.position)

            pygame.display.flip()


def main():
    pygame.init()
    pygame.display.set_caption("King's Table")
    screen = pygame.display.set_mode(WINDOW_SIZE)
    menu_items = ('Play as Attacker', 'Play as Defender', 'Play both sides', 'Computer vs Computer', 'Quit')
    gm = GameMenu(screen, menu_items)
    gm.run()
    print('You have selected: {}'.format(gm.selection))
    if not gm.selection or gm.selection == 'Quit':
        pass
    else:
        if gm.selection == 'Play as Attacker':
            attacker = True
            defender = False
        elif gm.selection == 'Play as Defender':
            attacker = False
            defender = True
        elif gm.selection == 'Play both sides':
            attacker = True
            defender = True
        else:
            attacker = False
            defender = False
        play = True

        while play:
            play = run_game(screen, attacker, defender)


if __name__ == "__main__":
    main()
