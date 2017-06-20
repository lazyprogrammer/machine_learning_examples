import numpy as np

STATE_SET = set()
LENGTH = 3


def get_state(board):
  # returns the current state, represented as an int
  # from 0...|S|-1, where S = set of all possible states
  # |S| = 3^(BOARD SIZE), since each cell can have 3 possible values - empty, x, o
  # some states are not possible, e.g. all cells are x, but we ignore that detail
  # this is like finding the integer represented by a base-3 number
  k = 0
  h = 0
  for i in xrange(LENGTH):
    for j in xrange(LENGTH):
      if board[i,j] == 0:
        v = 0
      elif board[i,j] == -1:
        v = 1
      elif board[i,j] == 1:
        v = 2
      h += (3**k) * v
      k += 1
  return h

def set_board(board, i=0, j=0):
  results = []

  for symbol in (0, -1, 1):
    board[i,j] = symbol

    if j == 2:
      # j goes back to 0, increase i, unless i = 2, then we are done
      if i == 2:
        # we are done!
        h = get_state(board)
        if h in STATE_SET:
          # bad!!!
          print "bad state:"
          print h
          print board
        assert(h not in STATE_SET)
        STATE_SET.add(h)
      else:
        results += set_board(board, i + 1, 0)
    else:
      results += set_board(board, i, j + 1)

  return results


board = np.zeros((3, 3))
set_board(board)
print len(STATE_SET)



