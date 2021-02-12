from functools import cached_property
from time import time

import settings as s
from fallbacks import pygame


class Item(object):
    def __init__(self):
        pass

    def avatar(self):
        raise NotImplementedError()

    def render(self, screen, x, y):
        screen.blit(self.avatar, (x, y))

    def get_state(self) -> tuple:
        raise NotImplementedError()


class Coin(Item):
    avatar = pygame.image.load('assets/coin.png')

    def __init__(self, pos, collectable=False):
        super(Coin, self).__init__()
        self.x = pos[0]
        self.y = pos[1]
        self.collectable = collectable

    def get_state(self):
        return self.x, self.y


class Bomb(Item):
    DEFAULT_AVATARS = {color: pygame.image.load(f'assets/bomb_{color}.png') for color in s.AGENT_COLORS}

    def __init__(self, pos, owner, timer, power, color, custom_sprite=None):
        super(Bomb, self).__init__()
        self.x = pos[0]
        self.y = pos[1]
        self.owner = owner
        self.timer = timer
        self.power = power

        self.active = True

        self.color = color
        self.custom_sprite = custom_sprite

    @cached_property
    def avatar(self):
        if self.custom_sprite:
            return self.custom_sprite
        return Bomb.DEFAULT_AVATARS[self.color]

    def get_state(self):
        return (self.x, self.y), self.timer

    def get_blast_coords(self, arena):
        x, y = self.x, self.y
        blast_coords = [(x, y)]

        for i in range(1, self.power + 1):
            if arena[x + i, y] == -1:
                break
            blast_coords.append((x + i, y))
        for i in range(1, self.power + 1):
            if arena[x - i, y] == -1:
                break
            blast_coords.append((x - i, y))
        for i in range(1, self.power + 1):
            if arena[x, y + i] == -1:
                break
            blast_coords.append((x, y + i))
        for i in range(1, self.power + 1):
            if arena[x, y - i] == -1:
                break
            blast_coords.append((x, y - i))

        return blast_coords


class Explosion(Item):
    STAGES = [pygame.image.load(f'assets/explosion_{i}.png') for i in range(6)]

    def __init__(self, blast_coords, screen_coords, owner, timer):
        super().__init__()
        self.blast_coords = blast_coords
        self.screen_coords = screen_coords
        self.owner = owner
        self.timer = timer
        self.active = True
        self.stages = Explosion.STAGES

    def render(self, screen, **kwargs):
        img = pygame.transform.rotate(self.stages[self.timer], (-50 * time()) % 360)
        rect = img.get_rect()
        for (x, y) in self.screen_coords:
            rect.center = x + 15, y + 15
            screen.blit(img, rect.topleft)
