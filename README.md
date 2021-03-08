# [markov walk](https://github.com/rtmigo/markov_walk#readme)
[![Actions Status](https://github.com/rtmigo/markov_walk/workflows/unit%20test/badge.svg?branch=master)](https://github.com/rtmigo/markov_walk/actions)
[![Generic badge](https://img.shields.io/badge/Python-3.8+-blue.svg)](#)

This module solves a particular mathematical problem related to probability theory. 

-----

Let's say a completely drunk passenger, while on a train, is trying to find his car. The cars are connected, 
so he wanders between them. Some transitions between cars attract him more, and some less.

At the very beginning and at the very end of the train there are ticket collectors: if they meet a drunkard, 
they will kick him out of the train.<sup id="a1">[*](#myfootnote1)</sup>

We know which car the drunkard is in now. The questions are:

- What is the probability that he will be thrown out by the ticket collector standing at the beginning of the train, and not at the end?

- What is the likelihood that he will at least visit his car before being kicked out?

# Stricter problem statement

We are dealing with a discrete 1D random walk. At each state, we have different probabilities of
making step to the left or to the right.

| States        |   L   |   0   |  < 1 >  |    2  |   3   |   4   |   5   |   R   |
|---------------|-------|-------|---------|-------|-------|-------|-------|-------|
| P(move right) |  ...  |**0.3**|    0.5  |**0.7**|  0.4  |  0.8  |  0.9  |  ...  |
| P(move left)  |  ...  |  0.7  |    0.5  |  0.3  |  0.6  |  0.2  |  0.1  |  ...  |

The probability to get from state `1` to state `2` is `0.7`.
 
The probability to get from state `1` to state `0` is `0.3`.

Suppose the motion begins at state `1`. How can we calculate the probability that we will get to state `R`
before we get to state `L`? What is the probability we will get to state `3` before `L` and `R`? 

# What the module does

It uses [Absorbing Markov chains](https://en.wikipedia.org/wiki/Absorbing_Markov_chain) to solve the problem.
It performs all matrix calculations in `numpy` and returns the answers as `float` numbers.

# How to install

```bash
cd /abc/your_project

# clone the module to /abc/your_project/markov_walk
svn export https://github.com/rtmigo/markov_walk/trunk/markov_walk markov_walk

# install dependencies
pip3 install numpy
``` 

Now you can `import markov_walk` from `/abc/your_project/your_module.py`

# How to use

```python3
from markov_walk import MarkovWalk

step_right_probs = [0.3, 0.5, 0.7, 0.4, 0.8, 0.9]
walk = MarkovWalk(step_right_probs)
```

- `ever_reach_probs[startPos][endPos]` is the probability, that after
infinite wandering started at `startPos` we will ever reach the point `endPos`

- `walk.right_edge_probs[pos]` is the probability for a starting point `pos`, that after infinite wandering we will leave 
the table on the right, and not on the left

By positions we mean indexes in `step_right_probs`. The point 1 from the example has index `0` in the list. So we will decrease all indices from the example by one.   

```python3
# the motion begins at state 2. 
# How can we calculate the probability that we will get to state R before we get to state L?
print(walk.right_edge_probs[2])

# the motion begins at state 1.
# What is the probability we will get to state 3 before L and R? 
print(walk.ever_reach_probs[1][3])

```
-----
<sup><a name="myfootnote1">*</a></sup> Perhaps you are worried about why the ticket collectors are going to kick the passenger out. The reason is that he is traveling on a lottery ticket.


