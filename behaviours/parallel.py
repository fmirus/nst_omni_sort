import nstbot
import numpy as np


periods = [2860, 4000, 5000, 6670]
freqs = 1000000 / np.array(periods, dtype=float)

use_bot = True

if not hasattr(nstbot, 'mybot'):
    if use_bot:
        bot = nstbot.OmniArmBot()
        bot.connect(nstbot.SocketList({'motors': ['10.162.177.29', 54322],
                                       'retina_left': ['10.162.177.29', 54320],
                                       'retina_right': ['10.162.177.29', 54321],
                                       'retina_arm': ['10.162.177.29', 54323],}))
        bot.tracker('retina_left', True, tracking_freqs=freqs, streaming_period=10000)
        bot.tracker('retina_right', True, tracking_freqs=freqs, streaming_period=10000)
        bot.tracker('retina_arm', True, tracking_freqs=freqs, streaming_period=10000)
    else:
        class DummyBot(object):
            def base_pos(self, x, msg_period):
                pass
            def arm(self, x, msg_period):
                pass
            def get_tracker_info(self, name, index):
                v = np.linspace(-0.8, 0.8, len(freqs))[index]
                return v,0,0,1
        bot = DummyBot()
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
                r = []
                for i in range(len(freqs)):
                    r.extend([
                        bot.get_tracker_info('retina_left', i),
                        bot.get_tracker_info('retina_right', i),
                        bot.get_tracker_info('retina_arm', i),
                        ])
                return np.hstack(r)

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
        nengo.Connection(botnet.tracker[24:36], self.info, synapse=None)


class OrientLR(nengo.Network):
    def __init__(self, target, botnet):
        super(OrientLR, self).__init__()
        #self.config[nengo.Ensemble].neuron_type = nengo.Direct()
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



class BehaviourControl(nengo.Network):
    def __init__(self, behaviours):
        super(BehaviourControl, self).__init__()
        with self:
            self.behave = nengo.Node([0]*len(behaviours))
        for i, b in enumerate(behaviours):
            nengo.Connection(self.behave[i], b.activation, synapse=None)


class Parallel(nengo.Network):
    def __init__(self, target, botnet):
        super(Parallel, self).__init__()
        self.config[nengo.Ensemble].neuron_type = nengo.Direct()

        with self:
            self.all_diff = nengo.networks.EnsembleArray(n_neurons=200,
                                    n_ensembles=len(freqs),
                                    ens_dimensions=2)

            self.all_diff.add_output('product', lambda x: x[0]*x[1])
            self.slide = nengo.Ensemble(n_neurons=200, dimensions=2)
            nengo.Connection(self.all_diff.product, self.slide[0],
                                transform=np.ones((1, len(freqs))))
            self.activation = nengo.Node(None, size_in=1)
            nengo.Connection(self.activation, self.slide[1], synapse=None)

        for i, ens in enumerate(self.all_diff.ensembles):
            nengo.Connection(botnet.tracker[12*i+0], ens[1], transform=0.5)
            nengo.Connection(botnet.tracker[12*i+4], ens[1], transform=0.5)

            nengo.Connection(botnet.tracker[12*i+0], ens[0], transform=1)
            nengo.Connection(botnet.tracker[12*i+4], ens[0], transform=-1)
            nengo.Connection(target.info[[0]], ens[0], transform=-1)
            nengo.Connection(target.info[[4]], ens[0], transform=1)


        nengo.Connection(self.slide, botnet.base_pos[1],
                         function=lambda x: x[0]*x[1],
                         transform=1, # change this to -1 to swap direction
                                      # increase value to slide faster
                         )



model = nengo.Network()
model.config[nengo.Ensemble].neuron_type=nengo.LIFRate()
with model:
    botnet = Bot(bot)
    target = TargetInfo(botnet)
    orient_lr = OrientLR(target, botnet)
    parallel = Parallel(target, botnet)

    bc = BehaviourControl([orient_lr, parallel])


