
try:
    import numpy as np
except:
    print('sorry to annoy :(, but for this shitty simulation we used the Numpy python library, you gotta install it '
          'if you want to go on ')

try:
    import pygame
except:
    print('sorry to annoy :(, but for this shitty simulation we used the PyGame python library, you gotta install it '
          'if you want to go on ')


######################################################################################################################
######################################################################################################################

print('\n\n---> RAWLAND SIMULATION <---\n\n'
      'each creature s aim is to seek and get food, as they swell up they also get faster. When they hit a threshold '
      'size, they undergo mitosis and give birth to children, some will have their DNA mutated while others will '
      'be a mere copy of their parents. Species will rise if the DNA is particularly unique (qualitatively different'
      'from other individuals). In red, Predators try to bite and suck energy from the food-seeking creatures; they '
      'are also subject to evolution. \n')

######################################################################################################################
######################################################################################################################


''' this is the Brain of each creature, 
-> it's a one-hidden-layer ANN  
-> the weights and biases are the genes of the DNA, which will evolve at each iteration '''


class Brain():

    def __init__(self, I, H, eta):
        O = 1
        self.wh = np.random.normal(0, 1 / np.sqrt(I), (H, I))
        self.wo = np.random.normal(0, 1 / np.sqrt(I), (O, H))
        self.bh = np.random.normal(0.3, 0, (H, 1))
        self.bo = np.random.normal(0.3, 0, (O, 1))
        self.eta = eta
        self.inodes = I
        self.energy = 5
        self.life = 5
        self.Hnodes = H

    def active(self, x, target):

        x /= 1000
        target /= 6.2832

        X = [x] * (self.inodes - 2)
        X.append(target)
        X.append(target)

        inputs = np.array(X, ndmin=2).T

        zh = np.dot(self.wh, inputs) + self.bh
        ah = sigmoid(zh)
        zo = np.dot(self.wo, ah) + self.bo
        ao = sigmoid(zo)

        if target:
            target = np.array(target, ndmin=2).T
            cost = np.array((ao - target) * sigmoid_prime(zo), ndmin=2)

            cost2 = np.dot(self.wo.T, cost) * sigmoid_prime(zh)

            cost2 = cost2[0][0]

            self.wh = self.wh - self.eta * np.dot(cost2.T, inputs.T)  # hidden weights

            self.bh = self.bh - self.eta * cost2

            self.wo = self.wo - self.eta * np.dot(cost, ah.T)  # output weights

            self.bo = self.bo - self.eta * cost  # output biases

        ao = ao.T
        return ao[0][0] * 6.2832

    def reproduce(self, color):

        copy_dna = [self.wh, self.wo, self.bh, self.bo]

        ## mutation

        wh_m, wo_m, bh_m = self.wh, self.wo, self.bo

        for i in range(2):
            wh_m[np.random.choice(range(len(wh_m)))] = np.random.normal(0, 0.5)
            wo_m[np.random.choice(range(len(wo_m)))] = np.random.normal(0, 0.5)

        mutated_dna = [wh_m, wo_m, bh_m]

        return copy_dna, mutated_dna, color


# sigmoid activation function and its derivative

def sigmoid(s):
    return 1 / (1 + np.exp(- s))


sigmoid = np.vectorize(sigmoid)


def sigmoid_prime(s):
    return sigmoid(s) * (1 - sigmoid(s))


sigmoid_prime = np.vectorize(sigmoid_prime)


# brain of the children, qualitatively very similar

class Brain_C():

    def __init__(self, I, H, eta, dna):

        wh, wo, bh = dna[0], dna[1], dna[2]

        O = 1
        self.wh = wh
        self.wo = wo
        self.bh = bh
        self.bo = np.random.normal(0.3, 0, (O, 1))
        self.eta = eta
        self.inodes = I
        self.Hnodes = H
        self.energy = 5
        self.life = 5

    def active(self, x, target):

        x /= 1000
        target /= 6.2832

        X = [x] * (self.inodes - 2)
        X.append(target)
        X.append(target)

        inputs = np.array(X, ndmin=2).T

        zh = np.dot(self.wh, inputs) + self.bh
        ah = sigmoid(zh)
        zo = np.dot(self.wo, ah) + self.bo
        ao = sigmoid(zo)

        if target:
            target = np.array(target, ndmin=2).T
            cost = np.array((ao - target) * sigmoid_prime(zo), ndmin=2)

            cost2 = np.dot(self.wo.T, cost) * sigmoid_prime(zh)

            cost2 = cost2[0][0]

            self.wh = self.wh - self.eta * np.dot(cost2.T, inputs.T)  # hidden weights

            self.bh = self.bh - self.eta * cost2

            self.wo = self.wo - self.eta * np.dot(cost, ah.T)  # output weights

            self.bo = self.bo - self.eta * cost  # output biases

        ao = ao.T
        return ao[0][0] * 6.2832

    def reproduce(self, color):

        copy_dna = [self.wh, self.wo, self.bh, self.bo]

        ## mutation

        wh_m, wo_m, bh_m = self.wh, self.wo, self.bo

        for i in range(2):
            wh_m[np.random.choice(range(len(wh_m)))] = np.random.normal(0, 0.5)
            wo_m[np.random.choice(range(len(wo_m)))] = np.random.normal(0, 0.5)

        mutated_dna = [wh_m, wo_m, bh_m]

        return copy_dna, mutated_dna, color


def sigmoid(s):
    return 1 / (1 + np.exp(- s))


sigmoid = np.vectorize(sigmoid)


def sigmoid_prime(s):
    return sigmoid(s) * (1 - sigmoid(s))


sigmoid_prime = np.vectorize(sigmoid_prime)



''' the Enviroment block '''

''' WITH GRAPHICS + '''

window_x, window_y = 1300, 750

n_creatures = int(input('\nsize of the food-seeking creatures population?\n'))
n_predators = int(input('\nsize of the predators population?\n'))

''' first generation '''

hidden_nodes = int(input('\nhidden neurons in the food-seeking creatures net?\n'))
population = []
chargespan = 10000
for u in range(n_creatures):
    brain = Brain(4, hidden_nodes, 1)
    body = pygame.Rect(window_x // 2, window_y // 2, 10, 10)
    speed = 5
    dx = speed * np.random.choice([-1, 1])
    dy = speed * np.random.choice([-1, 1])
    beta = np.pi / 4
    lifeclock = pygame.time.get_ticks()
    color = (np.random.choice(range(50, 120)), np.random.choice(range(50, 120)), np.random.choice(range(50, 120)))

    population.append([brain, body, speed, dx, dy, beta, lifeclock, color])

# predators
predators = []
chargespan_p = 4000
old_xv, old_yv = np.random.choice(range(100, window_x - 400)), np.random.choice(range(100, window_y - 200))
victim_targeted = False
old_target = 2
hidden_nodes_p = int( input('\nhidden neurons in the predators net ?\n'))

for u in range(n_predators):
    brain = Brain(4, hidden_nodes_p, 1)
    sight = 15
    body = pygame.Rect(window_x // 3, window_y // 3, 10, 10)
    speed = 7
    nv = np.random.choice(range(len(population)))

    dx = speed * np.random.choice([-1, 1])
    dy = speed * np.random.choice([-1, 1])
    beta = np.pi / 3
    huntingclock, eatingclock, lifeclock = pygame.time.get_ticks(), pygame.time.get_ticks(), pygame.time.get_ticks()
    color = (np.random.choice(range(100, 120)), np.random.choice(range(10, 30)), np.random.choice(range(10, 30)))

    predators.append([brain, body, speed, dx, dy, beta, lifeclock, color, sight, eatingclock, huntingclock, nv])

# food

xf, yf = window_x // 2, window_y // 2
food_dx, food_dy = 1, 1
fdirx, fdiry = 1, 1
food = [xf, yf]
ketax, ketay = 1, 1

krilltime = pygame.time.get_ticks()

# graphics win


count = 0
paraupdate = pygame.time.get_ticks()
people = []
for i in range(100):
    people.append(0)


''' CORE CODE '''

pygame.init()
env = pygame.display.set_mode((window_x, window_y))
pygame.display.set_caption('ENVIRONMENT')

run = True
while run:

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            run = False

    ####### first graphics
    people.append(len(population))

    if pygame.time.get_ticks() - paraupdate > 2000:
        # rectfill = Rectangle(Point(-2,140),Point(202, 202))
        # rectfill.setFill('white')
        # rectfill.setOutline('white')
        # rectfill.draw(win)
        # tpop = Text(Point(100, 160), 'Population {0}\nPredators {1}'.format(len(population), len(predators)))
        # tpop.setSize(15)
        # tpop.draw(win)
        # if count >= 200:
        # count = 0
        # rectfill2 = Rectangle(Point(-2,-2),Point(202, 140))
        # rectfill2.setFill('white')
        # rectfill2.setOutline('white')
        # rectfill2.draw(win)

        # Point(count, len(population) * 5).draw(win)

        count += 1

        # paraupdate = pygame.time.get_ticks()

        # plt.plot(range(99), people[-100: -1])
        # plt.show()

    ''' krill '''

    krill_where = False
    if pygame.time.get_ticks() - krilltime > 8000:
        ketax = 0
        ketay = 0
        xf, yf = np.random.choice(range(200, window_x - 400)), np.random.choice(range(200, window_y - 200))

        krilltime = pygame.time.get_ticks()
        krill_where = True

    food_dx, food_dy = 0, 0
    keys = pygame.key.get_pressed()

    if keys[pygame.K_LEFT] or keys[pygame.K_RIGHT] or keys[pygame.K_UP] or keys[pygame.K_DOWN] or keys[pygame.K_c]:

        if keys[pygame.K_LEFT]:
            food_dx = -15
        if keys[pygame.K_RIGHT]:
            food_dx = 15
        if keys[pygame.K_UP]:
            food_dy = -15
        if keys[pygame.K_DOWN]:
            food_dy = 15
        if keys[pygame.K_c]:
            xf, yf = window_x // 2, window_y // 2

        xf += food_dx
        yf += food_dy

    # ------------------------------------------------

    env.fill((0, 0, 0))  # Fills the screen with black +++++++++++++

    ''' generation '''

    color1 = (np.random.choice(range(100, 240)), np.random.choice(range(100, 240)), np.random.choice(range(100, 240)))
    pygame.draw.circle(env, color1, (xf, yf), 4)
    pygame.draw.line(env, (230, 230, 230), (953, 0), (953, window_y))

    deathlist = []
    liveslistR = []
    for creature, n in zip(population, range(len(population))):

        ## WHERE IS THE FOOD
        # angle
        if creature[1].centerx != xf and creature[1].centery != yf:
            if (creature[1].centerx - xf) < 0:
                alpha = np.pi + np.arctan((yf - creature[1].centery) / (creature[1].centerx - xf))
            elif (yf - creature[1].centery) < 0 and (creature[1].centerx - xf) > 0:
                alpha = 3 / 2 * np.pi - np.arctan((yf - creature[1].centery) / (creature[1].centerx - xf))
            else:
                alpha = np.arctan((yf - creature[1].centery) / (creature[1].centerx - xf))
        elif creature[1].centerx == xf:
            alpha = np.pi / 2
        elif creature[1].centery == yf:
            alpha = 0.1

        # radius
        radius = np.sqrt((yf - creature[1].centery) ** 2 + (creature[1].centerx - xf) ** 2)

        ## brain activity
        target_ang = alpha + np.pi
        if target_ang > np.pi * 2:
            target_ang -= np.pi * 2

        creature[5] = creature[0].active(radius, target_ang)  # steering angle

        creature[1].move_ip(creature[3] * np.cos(creature[5]), creature[4] * np.sin(creature[5]))  # move
        pygame.draw.circle(env, creature[7], (creature[1][0] + 5, creature[1][1] + 5),
                           7 + (creature[0].energy - 5))  # body
        pygame.draw.line(env, creature[7], (creature[1][0] + 5, creature[1][1] + 5), ((
        creature[1][0] + 5 + ((14 + (creature[0].energy - 5)) * np.cos(creature[5])),
        creature[1][1] + 5 - ((14 + (creature[0].energy - 5)) * np.sin(creature[5])))), 4)
        # mouth

        # hitting the wall
        if creature[1].centerx >= window_x - creature[2] - 350 or creature[1].centerx <= creature[2] + 5:
            creature[3] *= -1
        if creature[1].centery >= window_y - creature[2] - 5 or creature[1].centery <= creature[2] + 5:
            creature[4] *= -1

        # get the food
        if creature[1].collidepoint((xf, yf)):  ## eating

            creature[0].energy += 1
            creature[0].eta *= 0.7  # learning slower
            creature[1].inflate_ip(1, 1)  # real size increases
            speedup = np.random.choice(np.arange(1, 1.2, 0.1))
            creature[3] = min(creature[3] * speedup, 10)
            creature[4] = min(creature[4] * speedup, 10)

            # food change location
            xf, yf = np.random.choice(range(200, window_x - 400)), np.random.choice(range(200, window_y - 200))

            if xf >= window_x - 180 or xf <= 200:
                xf *= -1
            if yf >= window_y - 180 or yf <= 200:
                yf *= -1
            color1 = (
            np.random.choice(range(100, 240)), np.random.choice(range(100, 240)), np.random.choice(range(100, 240)))
            pygame.draw.circle(env, color1, (xf, yf), 4)  # food
            creature[6] = pygame.time.get_ticks()

        elif (pygame.time.get_ticks() - creature[6]) > chargespan:  ## starving
            creature[0].energy -= 1

            creature[6] = pygame.time.get_ticks()
        if creature[0].energy == 0:
            deathlist.append(n)
        else:
            liveslistR.append(creature[1])

        # reproduction

        if creature[0].energy > 8:
            dna_copied, dna_mutated, color = creature[0].reproduce(creature[7])
            if max(color) >= 200:
                print('change color!')
                colorpar = (
                np.random.choice(range(40, 110)), np.random.choice(range(40, 110)), np.random.choice(range(40, 110)))

            else:
                colorpar = (color[0] + 20, color[1] + 20, color[2] + 20)

            for u in range(3):
                brain = Brain_C(4, 30, 1, dna_copied)
                body = pygame.Rect(creature[1].centerx, creature[1].centery, 10, 10)
                speed = 6
                dx = speed * np.random.choice([-1, 1])
                dy = speed * np.random.choice([-1, 1])
                beta = np.pi / 4
                lifeclock = pygame.time.get_ticks()
                color = colorpar
                population.append([brain, body, speed, dx, dy, beta, lifeclock, color])

            for u in range(1):
                brain = Brain_C(4, 30, 1, dna_mutated)
                body = pygame.Rect(creature[1].centerx, creature[1].centery, 10, 10)
                speed = 6
                dx = speed * np.random.choice([-1, 1])
                dy = speed * np.random.choice([-1, 1])
                beta = np.pi / 4
                lifeclock = pygame.time.get_ticks()
                color = colorpar
                population.append([brain, body, speed, dx, dy, beta, lifeclock, color])

            creature[0].energy -= 6

    '''predators'''

    deathlist_p = []

    if len(liveslistR) == 0:  # if no prey is alive
        print('\n\n++ to much deaths ++\n\n')
        env.fill((120, 10, 40))  # Fills the screen with black +++++++++++++
        run = False

    else:

        for creature, n in zip(predators, range(len(predators))):

            if pygame.time.get_ticks() - creature[10] > 10000 or creature[11] >= len(liveslistR):
                creature[11] = np.random.choice([-1, 0])

                xv, yv = liveslistR[creature[11]].centerx, liveslistR[creature[11]].centery

                creature[10] = pygame.time.get_ticks()

            else:

                xv, yv = liveslistR[creature[11]].centerx, liveslistR[creature[11]].centery

            ## WHERE IS THE PREY
            # angle
            if creature[1].centerx != xv and creature[1].centery != yv:
                if (creature[1].centerx - xv) < 0:
                    alpha = np.pi + np.arctan((yv - creature[1].centery) / (creature[1].centerx - xv))
                elif (yv - creature[1].centery) < 0 and (creature[1].centerx - xv) > 0:
                    alpha = 3 / 2 * np.pi - np.arctan((yv - creature[1].centery) / (creature[1].centerx - xv))
                else:
                    alpha = np.arctan((yv - creature[1].centery) / (creature[1].centerx - xv))
            elif creature[1].centerx == xv:
                alpha = np.pi / 2
            elif creature[1].centery == yv:
                alpha = 0.1

            # radius
            radius = np.sqrt((yv - creature[1].centery) ** 2 + (creature[1].centerx - xv) ** 2)

            ## brain activity
            target_ang = alpha + np.pi
            if target_ang > np.pi * 2:
                target_ang -= np.pi * 2

            creature[5] = creature[0].active(radius, target_ang)  # steering angle

            creature[1].move_ip(creature[3] * np.cos(creature[5]), creature[4] * np.sin(creature[5]))  # move
            pygame.draw.circle(env, creature[7], (creature[1][0] + 5, creature[1][1] + 5),
                               7 + (creature[0].energy - 5))  # body
            pygame.draw.line(env, creature[7], (creature[1][0] + 5, creature[1][1] + 5), ((
            creature[1][0] + 5 + ((14 + (creature[0].energy - 5)) * np.cos(creature[5])),
            creature[1][1] + 5 - ((14 + (creature[0].energy - 5)) * np.sin(creature[5])))), 4)
            for u in range(5):
                jota1 = np.random.choice(np.arange(0, 6.28, 0.01))
                pygame.draw.line(env, (creature[7]), (creature[1][0] + 5, creature[1][1] + 5), ((
                creature[1][0] + 5 + ((creature[8]) * np.cos(jota1) - (creature[8]) * np.sin(jota1)),
                creature[1][1] + 5 + ((creature[8]) * np.sin(jota1) + (creature[8]) * np.cos(jota1)))))
            # mouth and sight

            # hitting the wall
            if creature[1].centerx >= window_x - creature[2] - 350 or creature[1].centerx <= creature[2] + 5:
                creature[3] *= -1
            if creature[1].centery >= window_y - creature[2] - 5 or creature[1].centery <= creature[2] + 5:
                creature[4] *= -1

            # get the food
            if creature[1].collidepoint((xv, yv)) and (pygame.time.get_ticks() - creature[9]) > 500:  ## eating
                creature[0].energy += 1
                creature[0].eta *= 1  # learning slower
                speedup = np.random.choice(np.arange(1, 1.3, 0.1))
                creature[3] = min(creature[3] * speedup, 8)
                creature[4] = min(creature[4] * speedup, 8)
                population[creature[11]][0].energy -= 1  # damage to the victim

                pygame.draw.line(env, (255, 255, 255), (xv, yv + 10), (xv, yv - 10), 3)
                pygame.draw.line(env, (255, 255, 255), (xv + 10, yv), (xv - 10, yv), 3)
                pygame.draw.line(env, (255, 255, 255), (xv + 6, yv + 6), (xv - 6, yv - 6), 3)
                pygame.draw.line(env, (255, 255, 255), (xv + 6, yv - 6), (xv - 16, yv - 6), 3)

                creature[6] = pygame.time.get_ticks()
                creature[9] = pygame.time.get_ticks()

            elif (pygame.time.get_ticks() - creature[6]) > chargespan_p:  ## starving
                creature[0].energy -= 1

                creature[6] = pygame.time.get_ticks()
                if creature[0].energy == 0:
                    deathlist_p.append(n)

            # reproduction

            if creature[0].energy > 10:
                dna_copied, dna_mutated, color = creature[0].reproduce(creature[7])
                if max(color) >= 230:
                    colopar = (np.random.choice(range(40, 110), np.random.choice(range(40, 110)),
                                                np.random.choice(range(40, 110))))
                else:
                    colorpar = (color[0] + 10, color[1] + 10, color[2] + 10)

                for u in range(1):
                    brain = Brain_C(4, 25, 1, dna_copied)
                    sight = 15
                    body = pygame.Rect(creature[1].centerx, creature[1].centery, 10, 10)
                    speed = 5
                    nv = np.random.choice(range(len(population)))

                    dx = speed * np.random.choice([-1, 1])
                    dy = speed * np.random.choice([-1, 1])
                    beta = np.pi / 3
                    huntingclock, eatingclock, lifeclock = pygame.time.get_ticks(), pygame.time.get_ticks(), pygame.time.get_ticks()
                    color = (
                    np.random.choice(range(100, 120)), np.random.choice(range(10, 30)), np.random.choice(range(10, 30)))

                    predators.append(
                        [brain, body, speed, dx, dy, beta, lifeclock, color, sight, eatingclock, huntingclock, nv])

                for u in range(0):
                    brain = Brain_C(4, 30, 1, dna_mutated)
                    sight = 15
                    body = pygame.Rect(creature[1].centerx, creature[1].centery, 10, 10)
                    speed = 5
                    nv = np.random.choice(range(len(population)))

                    dx = speed * np.random.choice([-1, 1])
                    dy = speed * np.random.choice([-1, 1])
                    beta = np.pi / 3
                    huntingclock, eatingclock, lifeclock = pygame.time.get_ticks(), pygame.time.get_ticks(), pygame.time.get_ticks()
                    color = (
                    np.random.choice(range(100, 120)), np.random.choice(range(10, 30)), np.random.choice(range(10, 30)))

                    predators.append(
                        [brain, body, speed, dx, dy, beta, lifeclock, color, sight, eatingclock, huntingclock, nv])

                creature[0].energy -= 9

        # prey update
        deathlist = []
        liveslistR = []
        for prey, n in zip(population, range(len(population))):
            if prey[0].energy == 0:
                deathlist.append(n)
            else:
                liveslistR.append(prey[1])

        deathlist.sort(reverse=True)
        for killed in deathlist:
            del population[killed]

        deathlist_p.sort(reverse=True)
        for killed in deathlist_p:
            del predators[killed]

    # show of the best

    if len(population) > 0:

        best_score = 1
        for runner, number in zip(population, range(len(population))):
            if runner[0].energy > best_score:
                best_one = number
        bt = False

        Hnod, weightsH, who = population[number][0].Hnodes, population[number][0].wh, population[number][0].wo
        weightsO = np.array(who, ndmin=2).T
        for wh, wo, s in zip(weightsH, weightsO, range(0, 430 + 430 // Hnod, 430 // Hnod)):
            wh1, wh2, wh3, wh4 = wh[0], wh[1], wh[2], wh[3]
            if np.sign(wh1) < 0:
                color = (255, 0, 0)
            else:
                color = (0, 255, 0)
            pygame.draw.line(env, color, (1000, 295), (1150, 15 + s), 1)

            if np.sign(wh2) < 0:
                color = (255, 0, 0)
            else:
                color = (0, 255, 0)
            pygame.draw.line(env, color, (1000, 255), (1150, 15 + s), 1)

            if np.sign(wh3) < 0:
                color = (255, 0, 0)
            else:
                color = (0, 255, 0)
            pygame.draw.line(env, color, (1000, 215), (1150, 15 + s), 1)

            if np.sign(wh4) < 0:
                color = (255, 0, 0)
            else:
                color = (0, 255, 0)
            pygame.draw.line(env, color, (1000, 175), (1150, 15 + s), 1)

            if np.sign(wo) < 0:
                color = (255, 0, 0)
            else:
                color = (0, 255, 0)
            pygame.draw.line(env, color, (1297, 215), (1150, 15 + s), 1)

            pygame.draw.circle(env, (250, 250, 240), (1150, 15 + s), 3, 3)  # hidden nodes

        pygame.draw.circle(env, (250, 250, 240), (1295, 215), 4, 3)  # output node

        pygame.draw.circle(env, (250, 250, 240), (1001, 295), 4, 3)  # input nodes
        pygame.draw.circle(env, (250, 250, 240), (1001, 255), 4, 3)
        pygame.draw.circle(env, (250, 250, 240), (1001, 215), 4, 3)
        pygame.draw.circle(env, (250, 250, 240), (1001, 175), 4, 3)

    # updating deaths

    # prey update
    deathlist = []
    liveslistR = []
    for prey, n in zip(population, range(len(population))):
        if prey[0].energy == 0:
            deathlist.append(n)
        else:
            liveslistR.append(prey[1])

    deathlist.sort(reverse=True)
    for killed in deathlist:
        del population[killed]

    deathlist_p = []
    for predator, m in zip(predators, range(len(predators))):
        if predator[0].energy <= 0:
            deathlist_p.append(m)

    deathlist_p.sort(reverse=True)
    for killed in deathlist_p:
        del predators[killed]

    if len(population) == 0:
        print('\n\n++ to much deaths ++\n\n')
        env.fill((120, 10, 40))  # Fills the screen with black +++++++++++++
        run = False

    pygame.display.update()

print('\n\n                                    G A M E  -  O V E R')
pygame.quit()

