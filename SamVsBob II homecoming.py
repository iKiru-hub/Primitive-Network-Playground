import numpy as np
import pygame


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



''' HOMECOMING '''


print('\n\nBob and Sam are now on the same page, they both have to get home. Unfortunately, Sam had a brain injury'
      ' and his brain is severely wounded, and smaller. You can move the home location, will they reach it?\n\n')

print('lets talk about hidden neurons (range 3-100)')
Hnod = int(input('how big is Bobs brain? '))
leaBob = float(input('how fast does Bob learn? range(0.001-2)'))
bob_limit = 30

leaSam = float(input('how about Sam? range(range 0.001-2)'))
sam_limit = 5

print('\n## lets talking about speed ##')
bob_speed = int(input('how fast should Bob go? (range 1-20)'))
sam_speed = int(input('and Sam? (range 1-20)'))

print('\nLETS GO HOME BITCHES\n')

window_x, window_y = 1200, 800

pygame.init()
env = pygame.display.set_mode((window_x, window_y))
pygame.display.set_caption('ENVIRONMENT')



''' initialization '''
bob_brain = Brain(5, Hnod, 1, leaBob)
bob = pygame.Rect(240, 240, 20, 20)

bob_dx = int(np.sqrt(bob_speed ** 2 // 2))
bob_dy = - int(np.sqrt(bob_speed ** 2 // 2)) + 1
beta = np.pi / 4

sam_brain = Brain(1, 4, 1, leaSam)
sam = pygame.Rect(400, 400, 20, 20)
sam_dx = int(np.sqrt(sam_speed ** 2 // 2))
sam_dy = - int(np.sqrt(sam_speed ** 2 // 2)) + 1
betas = np.pi / 4

# food
xf, yf = 200, 200
food_dx, food_dy = 0, 0
food = [xf, yf]

''' beef '''
run = True
while run:

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            run = False

    color1 = (np.random.choice(range(30, 240)), np.random.choice(range(30, 240)), np.random.choice(range(30, 240)))
    color2 = (np.random.choice(range(30, 240)), np.random.choice(range(30, 240)), np.random.choice(range(30, 240)))
    color3 = (np.random.choice(range(30, 240)), np.random.choice(range(30, 240)), np.random.choice(range(30, 240)))

    food_dx, food_dy = 0, 0
    keys = pygame.key.get_pressed()

    if keys[pygame.K_LEFT] or keys[pygame.K_RIGHT] or keys[pygame.K_UP] or keys[pygame.K_DOWN]:

        if keys[pygame.K_LEFT]:
            food_dx = -35
        if keys[pygame.K_RIGHT]:
            food_dx = 35
        if keys[pygame.K_UP]:
            food_dy = -35
        if keys[pygame.K_DOWN]:
            food_dy = 35

        xf += food_dx
        yf += food_dy

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
    if radius > bob_limit:
        bt = alpha + np.pi
        if bt > np.pi * 2:
            bt -= np.pi * 2
        beta, nothing, nothing2 = bob_brain.active(radius, bt)

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
    if sradius > bob_limit:
        bts = salpha + np.pi * 1.3
        if bts > np.pi * 2:
            bts -= np.pi * 2
        betas, nothings, nothings2 = sam_brain.active(sradius, bts)

    else:
        bts = False

    ''' rountine '''

    # hitting each other

    if 3 < 1:
        if bob.colliderect(sam):
            print('ahoheoho daamn why am I here')
        if bob.centerx + 10 <= sam.centerx + 10 or bob.centerx + 10 >= sam.centerx + 10:
            bob_dx *= -1
            sam_dx *= -1
        if bob.centery + 10 <= sam.centery + 10 or bob.centery + 10 >= sam.centery + 10:
            bob_dy *= -1
            sam_dy *= -1

    # hitting the frame
    if bob.centerx >= window_x - 10 or bob.centerx <= 10:
        bob_dx *= -1
    if bob.centery >= window_y - 10 or bob.centery <= 10:
        bob_dy *= -1

    if sam.centerx >= window_x - 10 or sam.centerx <= 10:
        sam_dx *= -1
    if sam.centery >= window_y - 10 or sam.centery <= 10:
        sam_dy *= -1

        # MOVEMENTS

    env.fill((0, 0, 0))  # Fills the screen with black

    bob.move_ip(bob_dx * np.cos(beta), bob_dy * np.sin(beta))  # BOB move
    pygame.draw.circle(env, (255, 0, 0), (bob[0], bob[1]), 10, 4)

    sam.move_ip(sam_dx * np.cos(betas), sam_dy * np.sin(betas))  # SAM move
    pygame.draw.rect(env, (0, 0, 255), sam, 4)

    # AREAS
    xhome, yhome = xf - 50, yf - 50

    pygame.draw.rect(env, color1, (xhome, yhome, 100, 100), 4)  # Bob area
    pygame.draw.polygon(env, color2, [[xhome, yhome], [xhome + 100, yhome], [xhome + 50, yhome - 50]], 4)
    pygame.draw.rect(env, color3, (xhome + 40, yhome + 75, 20, 25), 4)

    ''' brain display '''

    ## display brain Bob
    bt = False

    nothing, weightsH, weightsO = bob_brain.active(radius, bt)

    for wh, wo, s in zip(weightsH, weightsO, range(0, 430 + 430 // Hnod, 430 // Hnod)):
        wh1, wh2, wh3, wh4, wh5 = wh[0], wh[1], wh[2], wh[3], wh[4]
        if np.sign(wh1) < 0:
            color = (255, 0, 0)
        else:
            color = (0, 255, 0)
        pygame.draw.line(env, color, (900, 295), (1050, 15 + s), 1)

        if np.sign(wh2) < 0:
            color = (255, 0, 0)
        else:
            color = (0, 255, 0)
        pygame.draw.line(env, color, (900, 255), (1050, 15 + s), 1)

        if np.sign(wh3) < 0:
            color = (255, 0, 0)
        else:
            color = (0, 255, 0)
        pygame.draw.line(env, color, (900, 215), (1050, 15 + s), 1)

        if np.sign(wh4) < 0:
            color = (255, 0, 0)
        else:
            color = (0, 255, 0)
        pygame.draw.line(env, color, (900, 175), (1050, 15 + s), 1)

        if np.sign(wh5) < 0:
            color = (255, 0, 0)
        else:
            color = (0, 255, 0)
        pygame.draw.line(env, color, (900, 135), (1050, 15 + s), 1)

        if np.sign(wo) < 0:
            color = (255, 0, 0)
        else:
            color = (0, 255, 0)
        pygame.draw.line(env, color, (1195, 215), (1050, 15 + s), 1)

        pygame.draw.circle(env, (250, 250, 240), (1050, 15 + s), 3, 3)  # hidden nodes

    pygame.draw.circle(env, (250, 250, 240), (1195, 215), 4, 3)  # output node

    pygame.draw.circle(env, (250, 250, 240), (901, 295), 4, 3)  # input nodes
    pygame.draw.circle(env, (250, 250, 240), (901, 255), 4, 3)
    pygame.draw.circle(env, (250, 250, 240), (901, 215), 4, 3)
    pygame.draw.circle(env, (250, 250, 240), (901, 175), 4, 3)
    pygame.draw.circle(env, (250, 250, 240), (901, 135), 4, 3)

    ## display brain Sam
    bts = False

    nothings, weightsH, weightsO = sam_brain.active(sradius, bts)
    for wh, wo, s in zip(weightsH, weightsO, range(0, 80, 20)):
        wh1 = wh[0]
        if np.sign(wh1) < 0:
            color = (255, 0, 0)
        else:
            color = (0, 255, 0)
        pygame.draw.line(env, color, (900, 495), (1050, 465 + s), 1)

        if np.sign(wo) < 0:
            color = (255, 0, 0)
        else:
            color = (0, 255, 0)
        pygame.draw.line(env, color, (1195, 495), (1050, 465 + s), 1)

        pygame.draw.circle(env, (250, 250, 240), (1050, 465 + s), 3, 3)  # sam hiddens
    pygame.draw.circle(env, (250, 250, 240), (1195, 495), 4, 3)
    pygame.draw.circle(env, (250, 250, 240), (901, 495), 4, 3)

    pygame.draw.rect(env, (200, 0, 0), (855, 0, 460, 445), 1)  # bob brain outline
    pygame.draw.rect(env, (0, 0, 200), (855, 450, 460, 90), 1)  # sam brain outline

    if bob.centerx > window_x or bob.centery > window_y or sam.centerx > window_x or sam.centery > window_y or bob.centery < 0 or bob.centerx < 0 or sam.centery < 0 or sam.centerx < 0:
        print('############# !!! ALERT !!! #############')
        print('+++ ESCAPE +++')

    pygame.display.update()

pygame.quit()