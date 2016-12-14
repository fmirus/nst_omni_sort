import nstbot
import numpy as np

bot = nstbot.OmniArmBot()
bot.connect(nstbot.SocketList({'motors': ['10.162.177.29', 54322],
                               'retina_left': ['10.162.177.29', 54320]}))
bot.tracker('retina_left', True, tracking_freqs=[200], streaming_period=10000)
bot.send('motors', 'init_motors', '!G31610\n!G41170\n!G51276\n!G6210\n')


import nengo

model = nengo.Network()
with model:

    #def bot_base(t, x):
    #    bot.base_pos(x, msg_period=0.1)
    #    return x
    #base = nengo.Node(bot_base, size_in=3)
    #ctrl_base = nengo.Node([0,0,0])
    #nengo.Connection(ctrl_base, base)

    def bot_arm(t, x):
        offset = np.array([np.pi+1, np.pi-1.5, np.pi-1, 1])
        bot.arm(x+offset, msg_period=0.1)
        return x
    arm = nengo.Node(bot_arm, size_in=4)
    ctrl_arm = nengo.Node([0,0,0,0])
    nengo.Connection(ctrl_arm, arm)


    def bot_tracker(t):
        return bot.get_tracker_info('retina_left', 0)
        #return (bot.trk_px['retina_left'][0],
        #        bot.trk_py['retina_left'][0],
        #        bot.trk_radius['retina_left'][0],
        #        bot.trk_certainty['retina_left'][0])
    tracker = nengo.Node(bot_tracker)

#bot.arm([np.pi*scale, np.pi*scale, np.pi*scale, 0.5])



#bot.base([0.5, 0.5, 0.5])
#time.sleep(2)
#bot.base([0,0,0])
