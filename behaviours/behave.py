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
    bot.send('motors', 'init_motors_speed', '!P530\n')
    nstbot.mybot = bot
else:
    bot = nstbot.mybot

import nengo


class Bot(nengo.Network):
    def __init__(self, bot, msg_period=0.3,
                 arm_offset=[np.pi+1, np.pi-1.5, np.pi-1, 0]):
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
        
        with self:
            self.data = nengo.Ensemble(n_neurons=100, dimensions=2)
        nengo.Connection(target.info[[0, 4]], self.data)
        
        with self:
            self.spd = nengo.Ensemble(n_neurons=200, dimensions=2)
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
    def __init__(self, target, botnet, strength=2):
        super(ArmOrientLR, self).__init__()
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
                transform=-1)



class BehaviourControl(nengo.Network):
    def __init__(self, behaviours):
        super(BehaviourControl, self).__init__()
        with self:
            self.behave = nengo.Node([0]*len(behaviours))
        for i, b in enumerate(behaviours):
            nengo.Connection(self.behave[i], b.activation, synapse=None)


model = nengo.Network()
model.config[nengo.Ensemble].neuron_type=nengo.Direct()
with model:
    botnet = Bot(bot)
    target = TargetInfo(botnet)
    orient_lr = OrientLR(target, botnet)
    arm_orient_lr = ArmOrientLR(target, botnet)
    orient_fb = OrientFB(target, botnet)
    grasp_pos = GraspPosition(botnet)
    stay_away = StayAway(botnet)

    bc = BehaviourControl([orient_lr, arm_orient_lr, orient_fb, 
                           grasp_pos, stay_away])
