import nstbot
import numpy as np


# periods = [2860, 4000, 5000, 6670]
# freqs = 1000000 / np.array(periods, dtype=float)
periods = [2500, 2860, 4000, 5000]
freqs = np.ceil(1000000 / np.array(periods, dtype=float))


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


class OutOfOrder(nengo.Network):
    def __init__(self, botnet):
        super(OutOfOrder, self).__init__()
        with self:
            self.x_input = nengo.Node(None, size_in=len(freqs))

            self.diff = nengo.Ensemble(n_neurons=300, dimensions=len(freqs)-1)
            nengo.Connection(self.x_input[:-1], self.diff, transform=-1)
            nengo.Connection(self.x_input[1:], self.diff, transform=1)

            self.odd = nengo.Ensemble(n_neurons=400, dimensions=len(freqs))
            def detect_odd(x):
                mean = np.mean(x)
                x = x - mean
                score = np.zeros(len(freqs))
                score[:-1] += x
                score[1:] -= x
                score[0] *= 1.75
                score[-1] *= 1.75

                score = np.abs(score)
                worst = np.argmax(score)
                return np.eye(len(freqs))[worst]
            nengo.Connection(self.diff, self.odd, function=detect_odd)



        nengo.Connection(botnet.tracker[::12], self.x_input)







model = nengo.Network()
model.config[nengo.Ensemble].neuron_type=nengo.LIFRate()
with model:
    botnet = Bot(bot)
    order = OutOfOrder(botnet)
