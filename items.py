
import pygame
from pygame.locals import *
from pygame.transform import rotate
from time import time


class Item(object):

    def __init__(self):
        pass

    def render(self, screen, x, y):
        screen.blit(self.avatar, (x, y))


class Coin(Item):

    def __init__(self, pos):
        super(Coin, self).__init__()
        self.x = pos[0]
        self.y = pos[1]
        self.avatar = pygame.image.load('assets/coin.png')
        self.collectable = False

    def get_state(self):
        return (self.x, self.y)


class Bomb(Item):

    def __init__(self, pos, owner, timer, power, color, custom_sprite=None):
        super(Bomb, self).__init__()
        self.x = pos[0]
        self.y = pos[1]
        self.owner = owner
        self.timer = timer
        self.power = power

        if custom_sprite is None:
            self.avatar = pygame.image.load(f'assets/bomb_{color}.png')
        else:
            self.avatar = custom_sprite

        self.active = True

    def get_state(self):
        # return ((self.x, self.y), self.timer, self.power, self.active, self.owner.name)
        return (self.x, self.y, self.timer)

    def get_blast_coords(self, arena):
        x, y = self.x, self.y
        blast_coords = [(x,y)]

        for i in range(1, self.power+1):
            if arena[x+i,y] == -1: break
            blast_coords.append((x+i,y))
        for i in range(1, self.power+1):
            if arena[x-i,y] == -1: break
            blast_coords.append((x-i,y))
        for i in range(1, self.power+1):
            if arena[x,y+i] == -1: break
            blast_coords.append((x,y+i))
        for i in range(1, self.power+1):
            if arena[x,y-i] == -1: break
            blast_coords.append((x,y-i))

        return blast_coords


class Explosion(Item):

    def __init__(self, blast_coords, screen_coords, owner):
        self.blast_coords = blast_coords
        self.screen_coords = screen_coords
        self.owner = owner
        self.timer = owner.explosion_timer
        self.active = True

        self.stages = [pygame.image.load(f'assets/explosion_{i}.png') for i in range(6)]

    def render(self, screen):
        img = rotate(self.stages[self.timer], (-50*time()) % 360)
        rect = img.get_rect()
        for (x,y) in self.screen_coords:
            rect.center = x+15, y+15
            screen.blit(img, rect.topleft)
