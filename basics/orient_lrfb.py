import nstbot
import numpy as np

bot = nstbot.OmniArmBot()
bot.connect(nstbot.SocketList({'motors': ['10.162.177.29', 54322],
                               'retina_left': ['10.162.177.29', 54320],
                               'retina_right': ['10.162.177.29', 54321],
                               'retina_arm': ['10.162.177.29', 54323],}))
bot.tracker('retina_left', True, tracking_freqs=[200], streaming_period=10000)
bot.tracker('retina_right', True, tracking_freqs=[200], streaming_period=10000)
bot.tracker('retina_arm', True, tracking_freqs=[200], streaming_period=10000)
bot.send('motors', 'init_motors', '!G31610\n!G41170\n!G51276\n!G6210\n')


import nengo

model = nengo.Network()
model.config[nengo.Ensemble].neuron_type=nengo.Direct()
with model:

    def bot_base(t, x):
        bot.base_pos(x, msg_period=0.3)
        return x
    base = nengo.Node(bot_base, size_in=3)

    def bot_tracker(t):
        return np.hstack([
            bot.get_tracker_info('retina_left', 0),
            bot.get_tracker_info('retina_right', 0),
            bot.get_tracker_info('retina_arm', 0),
            ])

    tracker = nengo.Node(bot_tracker)


    x_data = nengo.Ensemble(n_neurons=100, dimensions=6, neuron_type=nengo.Direct())
    nengo.Connection(tracker[[0,3,4,7,8,11]], x_data)

    x_pos = nengo.Ensemble(n_neurons=100, dimensions=1, neuron_type=nengo.Direct())
    def compute_pos(x):
        lx, lc, rx, rc, ax, ac = x

        c = (lc + rc + ac)
        if c <= 0.1:
            return 0
        return (lx*lc+rx*rc+ax*ac) / c

    nengo.Connection(x_data, x_pos, function=compute_pos)

    nengo.Connection(x_pos, base[2], transform=-1)


    def x_vals(t, x):
        return x
    x = nengo.Node(x_vals, size_in=3)
    nengo.Connection(tracker[[0,4,8]], x)


    def c_vals(t, x):
        return x
    c = nengo.Node(c_vals, size_in=3)
    nengo.Connection(tracker[[3,7,11]], c)


    dist_data = nengo.Ensemble(n_neurons=100, dimensions=2)
    nengo.Connection(tracker[[0, 4]], dist_data)
    def dist_func(x):
        mid = x[0]+x[1]
        diff = x[0] - x[1]

        target_separation = 0.8
        if -0.5<mid<0.5:
            return (target_separation - diff)*10
        else:
            return 0
    nengo.Connection(dist_data, base[0], function=dist_func)


    def bot_arm(t, x):
        offset = np.array([np.pi-1.64, np.pi-2, np.pi+1.1, 1])
        bot.arm(x+offset, msg_period=0.2)
        return x
    arm = nengo.Node(bot_arm, size_in=4)
    ctrl_arm = nengo.Node([0,0,0,0])
    nengo.Connection(ctrl_arm, arm)


    ctrl_base = nengo.Node([-1,0,0])
    nengo.Connection(ctrl_base, base)
