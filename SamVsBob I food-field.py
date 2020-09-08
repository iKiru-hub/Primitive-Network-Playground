import numpy as np
import pygame



''' VOID CAPTION '''

print('\n\nThis is a naive food chase battle between Bob and Sam, both endowed with a simple MLP with one hidden layer.'
      ' Hunger lever are displayed in a bar. \nThey have a limited circular sensing visual field, resizable '
      'proportionally to their hunger. \n'
      'Their speed is also proportional to their hunger. \nYou can move the food with the arrows.\nThey can and will '
      'starve to death.\n\n')

print('Lets define how many hidden neurons each boy should have (range 3-100 advisable):')
Hnod = int(input('bob brain size? '))
leaBob = 1

Hnods = int(input('sam brain size? '))
leaSam = 1

bob_speed = 7
sam_speed = 7

print('\nFIGHT!\n')
print('\n\n## what is life after all ##')


window_x, window_y = 1300, 800

pygame.init()
env = pygame.display.set_mode((window_x, window_y))
pygame.display.set_caption('ENVIRONMENT')





''' the Neural Network '''


class Brain:
    def __init__(self, I, H, O, eta):
        self.wh = np.random.normal(0, 1 / np.sqrt(I), (H, I))
        self.wo = np.random.normal(0, 1 / np.sqrt(I), (O, H))
        self.bh = np.random.normal(0.3, 0, (H, 1))
        self.bo = np.random.normal(0.3, 0, (O, 1))
        self.eta = eta
        self.inodes = I
        self.energy = 5

    def active(self, x, target):
        x /= 1000
        target /= 6.2832

        inputs = np.array([x] * self.inodes, ndmin=2).T

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
        return ao[0][0] * 6.2832, self.wh, self.wo.T


def sigmoid(s):
    return 1 / (1 + np.exp(- s))


sigmoid = np.vectorize(sigmoid)


def sigmoid_prime(s):
    return sigmoid(s) * (1 - sigmoid(s))


sigmoid_prime = np.vectorize(sigmoid_prime)


''' initialization '''

bob_brain = Brain(5, Hnod, 1, leaBob)
bob = pygame.Rect(240, 240, 20, 20)
bob_dx = int(np.sqrt(bob_speed ** 2 // 2))
bob_dy = - int(np.sqrt(bob_speed ** 2 // 2)) + 1
beta = np.pi / 4
bob_sight = 200
bob_hunger = 10000
bob_hunwindow = 3000

sam_brain = Brain(5, Hnods, 1, leaSam)
sam = pygame.Rect(400, 400, 20, 20)
sam_dx = int(np.sqrt(sam_speed ** 2 // 2))
sam_dy = - int(np.sqrt(sam_speed ** 2 // 2)) + 1
betas = np.pi / 4
sam_sight = 200
sam_hunger = 10000
sam_hunwindow = 3000

# food

xf, yf = window_x // 2, window_y // 2
food_dx, food_dy = 1, 1
fdirx, fdiry = 1, 1
food = [xf, yf]
gotcha = False
ketax, ketay = 1, 1

bob_huntick, sam_huntick, krilltime, sam_meal, bob_meal = pygame.time.get_ticks(), pygame.time.get_ticks(), pygame.time.get_ticks(), pygame.time.get_ticks(), pygame.time.get_ticks()

''' beef '''
run = True
while run:

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            run = False

    ################################### Deus ex-machina ############################################################
    krill_where = False
    if pygame.time.get_ticks() - krilltime > 500:
        ketax = np.random.choice(np.arange(-1, 1, 0.05))
        ketay = np.random.choice(np.arange(-1, 1, 0.05))
        krilltime = pygame.time.get_ticks()
        krill_where = True

    food_dx, food_dy = 0, 0
    keys = pygame.key.get_pressed()

    if keys[pygame.K_LEFT] or keys[pygame.K_RIGHT] or keys[pygame.K_UP] or keys[pygame.K_DOWN]:

        if keys[pygame.K_LEFT]:
            food_dx = -20
        if keys[pygame.K_RIGHT]:
            food_dx = 20
        if keys[pygame.K_UP]:
            food_dy = -20
        if keys[pygame.K_DOWN]:
            food_dy = 20

        xf += food_dx
        yf += food_dy

    food_dx, food_dy = 3, 4
    xf += int(food_dx * ketax)
    yf += int(food_dy * ketay)

    ################################################################################################################

    if gotcha:
        xf, yf = np.random.choice(range(window_x // 3, window_x * 2 // 3)), np.random.choice(
            range(window_y // 3, window_y * 2 // 3))
        gotcha = False

    ''' BOB'''

    ## alpha
    if bob.centerx != xf and bob.centery != yf:
        if (bob.centerx - xf) < 0:  ##  -y --> pi + alpha
            alpha = np.pi + np.arctan((yf - bob.centery) / (bob.centerx - xf))
        elif (yf - bob.centery) < 0 and (bob.centerx - xf) > 0:
            alpha = 3 / 2 * np.pi - np.arctan((yf - bob.centery) / (bob.centerx - xf))
        else:
            alpha = np.arctan((yf - bob.centery) / (bob.centerx - xf))
    else:
        alpha = 1

    ## radius (polar coordinate)
    radius = np.sqrt((yf - bob.centery) ** 2 + (bob.centerx - xf) ** 2)

    ## brain activity
    if bob_sight > radius > 0:
        bt = alpha + np.pi
        if bt > np.pi * 2:
            bt -= np.pi * 2
        beta, nothing, nothing2 = bob_brain.active(radius, bt)

    elif pygame.time.get_ticks() - bob_huntick > bob_hunwindow:
        bt = False
        beta += np.random.choice([np.pi / 3, np.pi * 0.7, -np.pi / 3, -np.pi * 0.7, np.pi, np.pi / 2, -np.pi / 2])
        if beta > np.pi * 2:
            beta -= np.pi * 2
        bob_huntick = pygame.time.get_ticks()

    else:
        bt = False

    ''' SAM '''

    ## alpha
    if sam.centerx != xf and sam.centery != yf:
        if (sam.centerx - xf) < 0:  ##  -y --> pi + alpha
            salpha = np.pi + np.arctan((yf - sam.centery) / (sam.centerx - xf))
        elif (yf - sam.centery) < 0 and (sam.centerx - xf) > 0:
            salpha = 3 / 2 * np.pi - np.arctan((yf - sam.centery) / (sam.centerx - xf))
        else:
            salpha = np.arctan((yf - sam.centery) / (sam.centerx - xf))
    else:
        salpha = 1

    ## radius (polar coordinate)
    sradius = np.sqrt((yf - sam.centery) ** 2 + (sam.centerx - xf) ** 2)

    ## brain activity
    if sam_sight > sradius > 0:
        bts = salpha + np.pi * 1.3
        if bts > np.pi * 2:
            bts -= np.pi * 2
        betas, nothings, nothings2 = sam_brain.active(sradius, bts)


    elif pygame.time.get_ticks() - sam_huntick > sam_hunwindow:
        bt = False
        betas += np.random.choice([np.pi / 3, np.pi * 0.7, -np.pi / 3, -np.pi * 0.7, np.pi, np.pi / 2, -np.pi / 2])
        if betas > np.pi * 2:
            betas -= np.pi * 2
        sam_huntick = pygame.time.get_ticks()

    else:
        bts = False

    ''' ROUTINE '''

    # hitting each other
    if 3 < 4:
        if bob.colliderect(sam):
            if bob.centerx + 10 <= sam.centerx + 10 or bob.centerx + 10 >= sam.centerx + 10:
                bob_dx *= -1
                sam_dx *= -1
            if bob.centery + 10 <= sam.centery + 10 or bob.centery + 10 >= sam.centery + 10:
                bob_dy *= -1
                sam_dy *= -1

    # hitting the frame
    if bob.centerx >= window_x - max(bob_speed, sam_speed) - 3 or bob.centerx <= max(bob_speed, sam_speed) + 3:
        bob_dx *= -1
    if bob.centery >= window_y - max(bob_speed, sam_speed) - 3 or bob.centery <= max(bob_speed, sam_speed) + 3:
        bob_dy *= -1

    if sam.centerx >= window_x - max(bob_speed, sam_speed) - 3 or sam.centerx <= max(bob_speed, sam_speed) + 3:
        sam_dx *= -1
    if sam.centery >= window_y - max(bob_speed, sam_speed) - 3 or sam.centery <= max(bob_speed, sam_speed) + 3:
        sam_dy *= -1

    if xf >= window_x - 180 or xf <= 200:
        xf *= -1
    if yf >= window_y - 180 or yf <= 200:
        yf *= -1

        # MOVEMENTS

    env.fill((0, 0, 0))  # Fills the screen with black

    bob.move_ip(bob_dx * np.cos(beta) * (5 / bob_brain.energy),
                bob_dy * np.sin(beta) * (5 / bob_brain.energy))  # BOB move
    pygame.draw.circle(env, (255, 0, 0), (bob[0] + 10, bob[1] + 10), 10 + (bob_brain.energy - 5))  # body

    pygame.draw.circle(env, (5, 0, 0), (bob[0] + 10, bob[1] + 10), bob_sight, 1)  # bob sight
    for u in range(5):
        jota1 = np.random.choice(np.arange(0, 6.28, 0.01))
        pygame.draw.line(env, (150, 0, 0), (bob[0] + 10, bob[1] + 10),
                         ((bob[0] + 10 + ((bob_sight - 60) * np.cos(jota1) - (bob_sight - 60) * np.sin(jota1)),
                           bob[1] + 10 + ((bob_sight - 60) * np.sin(jota1) + (bob_sight - 60) * np.cos(jota1)))))

    pygame.draw.line(env, (255, 0, 0), (bob[0] + 10, bob[1] + 10),
                     ((bob[0] + 10 + (30 * np.cos(beta)), bob[1] + 10 - (30 * np.sin(beta)))), 7)
    # mouth

    sam.move_ip(sam_dx * np.cos(betas) * (5 / sam_brain.energy),
                sam_dy * np.sin(betas) * (5 / sam_brain.energy))  # SAM move
    pygame.draw.circle(env, (0, 0, 255), (sam[0] + 10, sam[1] + 10), 10 + (sam_brain.energy - 5))  # body

    pygame.draw.circle(env, (0, 0, 5), (sam[0] + 10, sam[1] + 10), sam_sight, 1)  # sam sight
    for u in range(5):
        jota1 = np.random.choice(np.arange(0, 6.28, 0.01))
        pygame.draw.line(env, (0, 0, 150), (sam[0] + 10, sam[1] + 10),
                         ((sam[0] + 10 + ((sam_sight - 60) * np.cos(jota1) - (sam_sight - 60) * np.sin(jota1)),
                           sam[1] + 10 + ((sam_sight - 60) * np.sin(jota1) + (sam_sight - 60) * np.cos(jota1)))))

    pygame.draw.line(env, (0, 0, 255), (sam[0] + 10, sam[1] + 10),
                     ((sam[0] + 10 + (30 * np.cos(betas)), sam[1] + 10 - (30 * np.sin(betas)))), 7)
    # mouth

    # FOOD

    color1 = (np.random.choice(range(100, 240)), np.random.choice(range(100, 240)), np.random.choice(range(100, 240)))

    pygame.draw.circle(env, color1, (xf, yf), 4)  # food

    if bob.collidepoint((xf, yf)):
        gotcha = True
        bob_brain.energy += 1
        bob_meal = pygame.time.get_ticks()
        bob_brain.eta += (5 - bob_brain.energy) * 0.1
        bob_hunwindow += 800
        bob_sight -= 10

    if sam.collidepoint((xf, yf)):
        gotcha = True
        sam_brain.energy += 1
        sam_meal = pygame.time.get_ticks()
        sam_brain.eta += (5 - sam_brain.energy) * 0.1
        sam_hunwindow += 800
        sam_sight -= 10

    if pygame.time.get_ticks() - bob_meal > bob_hunger:
        bob_brain.energy -= 1  # less energy
        bob_meal = pygame.time.get_ticks()
        bob_brain.eta /= (sigmoid(bob_brain.energy - 5) + 0.5)
        bob_hunwindow -= 900  # frenetic movement
        bob_sight += 10
        if bob_brain.eta > 2:
            bob_brain.eta = 2  # frenetic learning
        if bob_brain.energy == 0:
            print('\n\n ++ BOB DEAD ++')
            env.fill((0, 0, 170))  # Fills the screen with black
            run = False

    if pygame.time.get_ticks() - sam_meal > sam_hunger:
        sam_brain.energy -= 1
        sam_meal = pygame.time.get_ticks()
        sam_brain.eta /= (sigmoid(sam_brain.energy - 5) + 0.5)
        sam_hunwindow -= 900
        sam_sight += 20
        if sam_brain.eta > 2:
            sam_brain.eta = 2
        if sam_brain.energy == 0:
            print('\n\n ++ SAM DEAD ++')
            env.fill((170, 0, 0))  # Fills the screen with black
            run = False

    # getting fat making your less prone to learning

    ''' BRAIN DISPLAY '''

    ## display brain Bob
    bt = False

    nothing, weightsH, weightsO = bob_brain.active(radius, bt)

    for wh, wo, s in zip(weightsH, weightsO, range(0, 430 + 430 // Hnod, 430 // Hnod)):
        wh1, wh2, wh3, wh4, wh5 = wh[0], wh[1], wh[2], wh[3], wh[4]
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

        if np.sign(wh5) < 0:
            color = (255, 0, 0)
        else:
            color = (0, 255, 0)
        pygame.draw.line(env, color, (1000, 135), (1150, 15 + s), 1)

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
    pygame.draw.circle(env, (250, 250, 240), (1001, 135), 4, 3)

    ## display brain Sam
    bt = False

    nothing, weightsH, weightsO = sam_brain.active(sradius, bts)

    for wh, wo, s in zip(weightsH, weightsO, range(0, 430 + 430 // Hnods, 430 // Hnods)):
        wh1, wh2, wh3, wh4, wh5 = wh[0], wh[1], wh[2], wh[3], wh[4]
        if np.sign(wh1) < 0:
            color = (255, 0, 0)
        else:
            color = (0, 255, 0)
        pygame.draw.line(env, color, (2, 295), (150, 15 + s), 1)

        if np.sign(wh2) < 0:
            color = (255, 0, 0)
        else:
            color = (0, 255, 0)
        pygame.draw.line(env, color, (2, 255), (150, 15 + s), 1)

        if np.sign(wh3) < 0:
            color = (255, 0, 0)
        else:
            color = (0, 255, 0)
        pygame.draw.line(env, color, (2, 215), (150, 15 + s), 1)

        if np.sign(wh4) < 0:
            color = (255, 0, 0)
        else:
            color = (0, 255, 0)
        pygame.draw.line(env, color, (2, 175), (150, 15 + s), 1)

        if np.sign(wh5) < 0:
            color = (255, 0, 0)
        else:
            color = (0, 255, 0)
        pygame.draw.line(env, color, (2, 135), (150, 15 + s), 1)

        if np.sign(wo) < 0:
            color = (255, 0, 0)
        else:
            color = (0, 255, 0)
        pygame.draw.line(env, color, (295, 215), (150, 15 + s), 1)

        pygame.draw.circle(env, (250, 250, 240), (150, 15 + s), 3, 3)  # hidden nodes

    pygame.draw.circle(env, (250, 250, 240), (295, 215), 4, 3)  # output node

    pygame.draw.circle(env, (250, 250, 240), (3, 295), 4, 3)  # input nodes
    pygame.draw.circle(env, (250, 250, 240), (3, 255), 4, 3)
    pygame.draw.circle(env, (250, 250, 240), (3, 215), 4, 3)
    pygame.draw.circle(env, (250, 250, 240), (3, 175), 4, 3)
    pygame.draw.circle(env, (250, 250, 240), (3, 135), 4, 3)

    pygame.draw.rect(env, (200, 0, 0), (985, 0, 430, 445), 1)  # bob brain outline
    pygame.draw.line(env, (200, 0, 0), (window_x, 445), (window_x - bob_brain.energy * 10, 445), 10)  # life
    pygame.draw.rect(env, (0, 0, 200), (0, 0, 350, 445), 1)  # sam brain outline
    pygame.draw.line(env, (0, 0, 200), (0, 445), (sam_brain.energy * 10, 445), 10)  # life

    if bob.centerx > window_x + 20 or bob.centery > window_y + 20 or sam.centerx > window_x + 20 or sam.centery > window_y + 20 or bob.centery < - 20 or bob.centerx < - 20 or sam.centery < - 20 or sam.centerx < - 20:
        print()
        print('############# !!! ALERT !!! #############')
        print('\n+++ ESCAPE +++')

        for u in range(5):
            env.fill((200, 0, 0))  # Fills the screen with black
            env.fill((0, 200, 0))  # Fills the screen with black
            env.fill((0, 0, 200))  # Fills the screen with black

        run = False
    pygame.display.update()

print('\n\n                                    G A M E  -  O V E R')
pygame.quit()
