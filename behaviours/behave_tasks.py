import nstbot
import numpy as np

if not hasattr(nstbot, 'mybot'):
    bot = nstbot.OmniArmBot()
    bot.connect(nstbot.SocketList({'motors': ['10.162.177.29', 54322],
                                   'retina_left': ['10.162.177.29', 54320],
                                   'retina_right': ['10.162.177.29', 54321],
                                   'retina_arm': ['10.162.177.29', 54323],}))
    bot.tracker('retina_left', True, tracking_freqs=[200], streaming_period=10000)
    bot.tracker('retina_right', True, tracking_freqs=[200], streaming_period=10000)
    bot.tracker('retina_arm', True, tracking_freqs=[200], streaming_period=10000)
    bot.send('motors', 'init_motors', '!G31610\n!G41170\n!G51276\n!G6210\n')
    bot.send('motors', 'init_motors_speed', '!P520\n')
    nstbot.mybot = bot
else:
    bot = nstbot.mybot

import nengo


class Bot(nengo.Network):
    def __init__(self, bot, msg_period=0.3,
                 arm_offset=[np.pi+1, np.pi-1.5, np.pi-1, 1]):
        super(Bot, self).__init__()
        self.arm_offset = np.array(arm_offset)
        with self:
            def bot_base(t, x):
                bot.base_pos(x, msg_period=msg_period)
                return x
            self.base_pos = nengo.Node(bot_base, size_in=3)

            def bot_tracker(t):
                return np.hstack([
                    bot.get_tracker_info('retina_left', 0),
                    bot.get_tracker_info('retina_right', 0),
                    bot.get_tracker_info('retina_arm', 0),
                    ])

            self.tracker = nengo.Node(bot_tracker)


            def bot_arm(t, x):
                bot.arm(x+self.arm_offset, msg_period=msg_period)
                return x
            self.arm = nengo.Node(bot_arm, size_in=4)

            self.ctrl_arm = nengo.Node([0,0,0,0])
            nengo.Connection(self.ctrl_arm, self.arm, synapse=None)

            self.ctrl_base = nengo.Node([0,0,0])
            nengo.Connection(self.ctrl_base, self.base_pos, synapse=None)

class TargetInfo(nengo.Network):
    def __init__(self, botnet):
        super(TargetInfo, self).__init__()
        with self:
            self.info = nengo.Node(None, size_in=12)
        nengo.Connection(botnet.tracker, self.info, synapse=None)

class OrientLR(nengo.Network):
    def __init__(self, target, botnet):
        super(OrientLR, self).__init__()
        self.config[nengo.Ensemble].neuron_type = nengo.LIFRate()
        with self:
            self.x_data = nengo.Ensemble(n_neurons=500, dimensions=6, radius=3)
        nengo.Connection(target.info[[0,3,4,7,8,11]], self.x_data)

        with self:
            self.x_pos = nengo.Ensemble(n_neurons=200, dimensions=2)
            def compute_pos(x):
                lx, lc, rx, rc, ax, ac = x

                c = (lc + rc + ac)
                if c <= 0.1:
                    return 0
                return (lx*lc+rx*rc+ax*ac) / c

            nengo.Connection(self.x_data, self.x_pos[0], function=compute_pos)

            self.activation = nengo.Node(None, size_in=1)
            nengo.Connection(self.activation, self.x_pos[1], synapse=None)

        nengo.Connection(self.x_pos, botnet.base_pos[2],
                         function=lambda x: x[0]*x[1],
                         transform=-1)


class OrientFB(nengo.Network):
    def __init__(self, target, botnet):
        super(OrientFB, self).__init__()
        self.config[nengo.Ensemble].neuron_type = nengo.LIFRate()

        with self:
            self.data = nengo.Ensemble(n_neurons=500, dimensions=2, radius=1.5)
        nengo.Connection(target.info[[0, 4]], self.data)

        with self:
            self.spd = nengo.Ensemble(n_neurons=500, dimensions=2, radius=1.5)
            def dist_func(x):
                mid = x[0]+x[1]
                diff = x[0] - x[1]

                target_separation = 0.8
                if -0.5<mid<0.5:
                    return (target_separation - diff)*10
                else:
                    return 0
            nengo.Connection(self.data, self.spd[0], function=dist_func)


            self.activation = nengo.Node(None, size_in=1)
            nengo.Connection(self.activation, self.spd[1], synapse=None)

        nengo.Connection(self.spd, botnet.base_pos[0],
                         function=lambda x: x[0]*x[1])

class GraspPosition(nengo.Network):
    def __init__(self, botnet):
        super(GraspPosition, self).__init__()
        with self:
            self.activation = nengo.Node(None, size_in=1)
        nengo.Connection(self.activation, botnet.arm,
                transform=[[-1.86], [-0.28], [2.10], [0]])


class ArmOrientLR(nengo.Network):
    def __init__(self, target, botnet, strength=1):
        super(ArmOrientLR, self).__init__()
        self.config[nengo.Ensemble].neuron_type = nengo.LIFRate()
        with self:
            self.x_data = nengo.Ensemble(n_neurons=500, dimensions=2, radius=3)
        nengo.Connection(target.info[[8,11]], self.x_data)

        with self:
            self.x_pos = nengo.Ensemble(n_neurons=200, dimensions=2)
            def compute_pos(x):
                x, c = x

                if c <= 0.1:
                    return 0
                return -x

            nengo.Connection(self.x_data, self.x_pos[0], function=compute_pos)

            self.activation = nengo.Node(None, size_in=1)
            nengo.Connection(self.activation, self.x_pos[1], synapse=None)

        nengo.Connection(self.x_pos, botnet.base_pos[2],
                         function=lambda x: x[0]*x[1],
                         transform=strength)


class StayAway(nengo.Network):
    def __init__(self, botnet):
        super(StayAway, self).__init__()
        with self:
            self.activation = nengo.Node(None, size_in=1)
        nengo.Connection(self.activation, botnet.base_pos[0],
                transform=-1.5)

class Grip(nengo.Network):
    def __init__(self, botnet):
        super(Grip, self).__init__()
        with self:
            self.activation = nengo.Node(None, size_in=1)
        nengo.Connection(self.activation, botnet.arm[3],
                transform=-0.4)



class BehaviourControl(nengo.Network):
    def __init__(self, behaviours):
        super(BehaviourControl, self).__init__()
        with self:
            self.behave = nengo.Node([0]*len(behaviours))
        for i, b in enumerate(behaviours):
            nengo.Connection(self.behave[i], b.activation, synapse=None)


class TaskGrab(nengo.Network):
    def __init__(self, target, botnet, behaviours, grabbed):
        super(TaskGrab, self).__init__()
        with self:
            self.activation = nengo.Node(None, size_in=1)
            self.behave = nengo.networks.EnsembleArray(n_neurons=400,
                                n_ensembles=6, ens_dimensions=2, radius=1.5,
                                neuron_type=nengo.LIFRate())
            self.behave.add_output('scaled', function=lambda x: x[0]*x[1])
            for i in range(len(behaviours)):
                nengo.Connection(self.activation, self.behave.ensembles[i][1],
                                 synapse=None)

            self.everything = nengo.Ensemble(n_neurons=100, dimensions=12,
                                    neuron_type=nengo.Direct())
            def do_it(x):
                lx,ly,lr,lc, rx,ry,rr,rc, ax,ay,ar,ac = x

                ORIENT_LR = 0
                ARM_ORIENT_LR = 1
                ORIENT_FB = 2
                GRASP_POS = 3
                STAY_AWAY = 4
                GRIP = 5

                diff = lx - rx

                result = [0,0,0,0,0,0]
                if (lc < 0.5 or rc < 0.5) and ac < 0.1:
                    result[STAY_AWAY] = 1
                else:
                    result[ORIENT_FB] = 1
                    result[GRASP_POS] = 1
                    if ac > 0.1:
                        result[ARM_ORIENT_LR] = 1
                    else:
                        result[ORIENT_LR] = 1
                        result[STAY_AWAY] = 1
                        if diff > 0.75:   # if we are too close
                            result[GRASP_POS] = 0

                return result
            nengo.Connection(self.everything, self.behave.input[::2],
                             function=do_it)

            self.should_close = nengo.Ensemble(n_neurons=200, dimensions=1,
                                   neuron_type=nengo.LIFRate(), radius=1.2)
            def do_should_close(x):
                result = 0
                lx,ly,lr,lc, rx,ry,rr,rc, ax,ay,ar,ac = x
                diff = lx - rx
                mid = (lx + rx) / 2
                if lc > 0.5 and rc > 0.5 and np.abs(diff-0.7)<0.03 and np.abs(mid)<0.05:
                    result = 1
                return result
            nengo.Connection(self.everything, self.should_close, function=do_should_close,
                             synapse=None)
            nengo.Connection(self.should_close, self.behave.input[10], synapse=0.2)
        nengo.Connection(self.should_close, grabbed.has_grabbed,
                         synapse=0.1, transform=2)

        nengo.Connection(target.info, self.everything, synapse=None)

        for i, b in enumerate(behaviours):
            nengo.Connection(self.behave.scaled[i], b.activation, synapse=None)



class TaskHold(nengo.Network):
    def __init__(self, grip):
        super(TaskHold, self).__init__()
        with self:
            self.activation = nengo.Node(None, size_in=1)
        nengo.Connection(self.activation, grip.activation, transform=1)



class Grabbed(nengo.Network):
    def __init__(self, botnet):
        super(Grabbed, self).__init__()
        with self:
            self.has_grabbed = nengo.Ensemble(n_neurons=50, dimensions=1,
                                              neuron_type=nengo.LIFRate())
            def state(x):
                if x<0.5: return 0
                else: return 1
            nengo.Connection(self.has_grabbed, self.has_grabbed, synapse=0.1)
        def opened_gripper(x):
            if x > -0.1:
                return -1
            else:
                return 0
        nengo.Connection(botnet.arm[3], self.has_grabbed,
                         function=opened_gripper)

class TaskGrabAndHold(nengo.Network):
    def __init__(self, task_grab, task_hold, grabbed):
        super(TaskGrabAndHold, self).__init__()
        self.config[nengo.Ensemble].neuron_type = nengo.LIFRate()

        with self:
            self.activation = nengo.Node(None, size_in=1)

            self.choice = nengo.Ensemble(n_neurons=300, dimensions=2, radius=1.5)
            nengo.Connection(self.activation, self.choice[0])
        nengo.Connection(grabbed.has_grabbed, self.choice[1])
        def choose_grab(x):
            if x[1] < 0.5 and x[0]>0.5:
                return 1
            else:
                return 0
        nengo.Connection(self.choice, task_grab.activation, function=choose_grab)
        def choose_hold(x):
            if x[1] > 0.5 and x[0]>0.5:
                return 1
            else:
                return 0
        nengo.Connection(self.choice, task_hold.activation, function=choose_hold)





model = nengo.Network(seed=2)
with model:
    botnet = Bot(bot)
    target = TargetInfo(botnet)
    orient_lr = OrientLR(target, botnet)
    arm_orient_lr = ArmOrientLR(target, botnet)
    orient_fb = OrientFB(target, botnet)
    grasp_pos = GraspPosition(botnet)
    stay_away = StayAway(botnet)
    grip = Grip(botnet)

    grabbed = Grabbed(botnet)

    task_grab = TaskGrab(target, botnet, [orient_lr, arm_orient_lr, orient_fb,
                           grasp_pos, stay_away, grip], grabbed)

    task_hold = TaskHold(grip)

    task_grab_and_hold = TaskGrabAndHold(task_grab, task_hold, grabbed)

    bc = BehaviourControl([orient_lr, arm_orient_lr, orient_fb,
                           grasp_pos, stay_away, grip, task_grab, task_hold,
                           task_grab_and_hold])


if __name__ == '__main__':
    with model:
        start = nengo.Node([1])
        nengo.Connection(start, task_grab_and_hold.activation)
    sim = nengo.Simulator(model)
    while True:
        sim.run(10)
