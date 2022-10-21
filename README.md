# Bloxorz #

![alt text](http://jpg.hoodamath.com/large/bloxorz_300x225.jpg)

Bloxorz is a game where the goal is to drop a 1×2×1 block through a hole in the middle of a board without falling off its edges. This game is available at
http://www.coolmath-games.com/0-bloxorz/index.html

## Definition of the game ##

Board is represented as a matrix:
```
[
    ['O', 'O', 'O', 'O', 'X', 'X', 'X', 'X', 'X', 'X'],
    ['O', 'O', 'O', 'O', 'X', 'X', 'X', 'O', 'O', 'O'],
    ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'G'],
    ['O', 'O', 'O', 'O', 'X', 'X', 'X', 'O', 'O', 'O'],
    ['O', 'O', 'O', 'O', 'X', 'X', 'X', 'X', 'X', 'X'],
    ['S', 'S', 'O', 'O', 'X', 'X', 'X', 'X', 'X', 'X']
]
```

* O denotes safe tiles: the block can stand on these anytime
* X denotes empty tiles: the block may never touch an empty tile, even if half of the block is on a safe tile
* S denotes the tile(s) occupied by the block: if the block is in the vertical orientation then there is one tile labeled S, otherwise (if the block is in the horizontal orientation) there are two adjacent tiles labeled S
* G denotes the goal tile: the block needs to be on it (vertically) in order to fall into the goal

### Solution ###
Genetic algorithm

### Credit ###
Ege Alpay's solutions for this problem are on his [Gitlab repo](https://gitlab.com/egealpay/bloxorz-solver).
