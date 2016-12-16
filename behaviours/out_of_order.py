import nstbot
import numpy as np


periods = [2860, 4000, 5000, 6670]
freqs = 1000000 / np.array(periods, dtype=float)

if not hasattr(nstbot, 'mybot'):
    bot = nstbot.OmniArmBot()
    bot.connect(nstbot.SocketList({'motors': ['10.162.177.29', 54322],
                                   'retina_left': ['10.162.177.29', 54320],
                                   'retina_right': ['10.162.177.29', 54321],
                                   'retina_arm': ['10.162.177.29', 54323],}))
    bot.tracker('retina_left', True, tracking_freqs=freqs, streaming_period=10000)
    bot.tracker('retina_right', True, tracking_freqs=freqs, streaming_period=10000)
    bot.tracker('retina_arm', True, tracking_freqs=freqs, streaming_period=10000)
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
            self.certainty = nengo.Ensemble(n_neurons=100, dimensions=len(freqs))
            
            rng = np.random.RandomState(seed=1)
            pts = rng.uniform(-1, 1, size=(1000,len(freqs)))
            
            self.pos_x = nengo.Ensemble(n_neurons=500, dimensions=len(freqs), radius=2,
                                    eval_points=pts,
                                    neuron_type=nengo.Direct())
            

                
            def detect(x):
                minx = min(x)
                maxx = max(x)
                
                ideal = np.linspace(minx, maxx, len(x))
                delta = np.abs(x - ideal)
                
                worst = np.argmax(delta)
                return np.eye(len(x))[worst]

                
        
            
            
            self.score = nengo.Ensemble(n_neurons=100, dimensions=len(freqs))
            nengo.Connection(self.pos_x, self.score, function=detect)

            def okay_func(x):
                order = np.argsort(x)
                if np.all(np.sort(order) == order):
                    return 1
                if np.all(np.sort(order) == order[::-1]):
                    return 1
                return 0
                
            
                


            
            self.okay = nengo.Ensemble(n_neurons=50, dimensions=1)
            nengo.Connection(self.pos_x, self.okay, function=okay_func)
            
        nengo.Connection(botnet.tracker[::12], self.pos_x)
        nengo.Connection(botnet.tracker[3::12], self.certainty)
        
        
        
        
        
        

model = nengo.Network()
model.config[nengo.Ensemble].neuron_type=nengo.Direct()
with model:
    botnet = Bot(bot)
    order = OutOfOrder(botnet)
