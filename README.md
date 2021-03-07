# [markov walk](https://github.com/rtmigo/markov_walk#readme)
[![Actions Status](https://github.com/rtmigo/markov_walk/workflows/unit%20test/badge.svg?branch=master)](https://github.com/rtmigo/vien/actions)
[![Generic badge](https://img.shields.io/badge/Python-3.8+-blue.svg)](#)

This module solves a particular mathematical problem related to probability theory. 

----

# The problem

Suppose we are dealing with a discrete 1D random walk. At each point, we have different probabilities of
making step to the left or to the right.


|Points        | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|--------------|---|---|---|---|---|---|---|---|
|P(move right) |...|0.3|0.5|0.7|0.4|0.8|0.9|...|
|P(move left)  |...|0.7|0.5|0.3|0.6|0.2|0.1|...|


For example, the probability to get from point 3 to point 4 is 0.7, and the probability to get from same
point 3 to 2 is 0.3.

In other words, it is like a Markov chain: states are points; transitions are possible only between
neighboring states; all transition probabilities are known.

Suppose the motion begins at point 3. How can we calculate the probability that we will get to point 7
before we get to point 0?

## The solution

The question was asked on StackExchange and got the [answer](https://math.stackexchange.com/a/2912626) from Aaron Montgomery: 

> The book "Random Walks and Electric Networks" has some useful examples that should be of assistance:
> https://math.dartmouth.edu/~doyle/docs/walks/walks.pdf
> In particular, I'll point you to section 1.2.6 -- particularly, the part starting with, "As a second example,"
> on the top of page 26.

# The code

The solution was implemented in module `markov_walk`.

It is not intended to be released as a package, so there is no installation instructions.

To solve the problem described:

```python3
from markov_walk import MarkovWalk

step_right_probs = [0.3, 0.5, 0.7, 0.4, 0.8, 0.9]
walk = MarkovWalk(step_right_probs)
```

So now

`ever_reach_probs[startPos][endPos]` is the probability, that after
infinite wandering started at `startPos` will will ever reach the point `endPos`.

`walk.right_edge_probs[pos]` is the probability for a starting point `pos`, that after infinite wandering we will leave 
the table on the right, and not on the left.

