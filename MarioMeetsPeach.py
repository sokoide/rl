import numpy as np
WIDTH = 6
HEIGHT = 6


class Env():
    def __init__(self):
        self.reset()

    def reset(self):
        state = [0, 0]  # x, y
        # make a maze
        m = 'M000B0,0000B0,0000B0,0000B0,000000,00000P'
        self._maze = np.empty(WIDTH * HEIGHT)
        i = 0
        for c in m:
            if c == ',':
                continue
            cell = 0
            if c == 'M':
                cell = 1
            if c == 'B':
                cell = 2
            if c == 'P':
                cell = 3
            self._maze[i] = cell
            i += 1
        self._maze = self._maze.reshape(WIDTH, HEIGHT)
        return state

    def step(self, state, action):
        # action: 0 -> right, 1-> down
        reward = -1
        done = False
        x, y = state[0], state[1]

        if action == 0 and x + 1 < WIDTH:
            x += 1
        elif action == 1 and y + 1 < HEIGHT:
            y += 1
        else:
            reward = -10  # can't move
            done = True
            return [x, y], reward, done

        if self._maze[y][x] == 3:
            # Peach
            print('x:{}, y:{}, met Peach'.format(x, y))
            reward = 5
            done = True
        elif self._maze[y][x] == 2:
            # Bower
            reward = -5

        return [x, y], reward, done


class Q():
    def __init__(self):
        self._Q = np.zeros((WIDTH, HEIGHT, 2))

    def get_action(self, state):
        if np.random.uniform(0, 1) > 0.5:
            next_action = np.random.choice([0, 1])
        else:
            x, y = state[0], state[1]
            a = np.where(self._Q[y][x] == self._Q[y][x].max())[0]
            next_action = np.random.choice(a)  # 'a' can be multiple
        return next_action

    def update(self, state, action, reward, next_state):
        alpha = 0.5
        gamma = 0.9
        next_max_q = max(self._Q[next_state[1]][next_state[0]])

        x, y = state[0], state[1]

        self._Q[y][x][action] = (
            1 - alpha) * self._Q[y][x][action] + alpha * (reward + gamma * next_max_q)

    def dump(self):
        print('    ', end='')
        for x in range(WIDTH):
            print('{:7}'.format(x), end='')
        print()
        r = 0
        for row in self._Q:
            right = ''
            down = ''
            for col in row:
                # print('→{:+.2f},↓{:+.2f} '.format(col[0], col[1]), end='')
                right += '{:>6.2f} '.format(col[0])
                down += '{:>6.2f} '.format(col[1])
            print('{:4d} →: {}'.format(r, right))
            print('     ↓: {}'.format(down))
            r += 1


def main():
    env = Env()
    q = Q()
    for episode in range(1, 101):
        state = env.reset()
        for t in range(10_000):
            action = q.get_action(state)
            next_state, reward, done = env.step(state, action)
            q.update(state, action, reward, next_state)
            state = next_state
            if done:
                break
        print('* episode: {}'.format(episode))
        q.dump()


main()
