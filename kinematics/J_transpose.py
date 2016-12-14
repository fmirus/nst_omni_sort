import nengo
import numpy as np

x0 = 0
y0 = -125

l1 = 156
l2 = 142
l34 = 155

def jacobian(q):
    return np.array([
        [l1*np.cos(q[0])+l2*np.cos(q[0]+q[1])+l34*np.cos(q[0]+q[1]+q[2]),
         l2*np.cos(q[0]+q[1])+l34*np.cos(q[0]+q[1]+q[2]),
         l34*np.cos(q[0]+q[1]+q[2])],
        [l1*np.sin(q[0])+l2*np.sin(q[0]+q[1])+l34*np.sin(q[0]+q[1]+q[2]),
         l2*np.sin(q[0]+q[1])+l34*np.sin(q[0]+q[1]+q[2]),
         l34*np.sin(q[0]+q[1]+q[2])],
         ])
        
def fwd(q):
    return np.array([
        l1*np.sin(q[0])+l2*np.sin(q[0]+q[1])+l34*np.sin(q[0]+q[1]+q[2]),
        -(l1*np.cos(q[0])+l2*np.cos(q[0]+q[1])+l34*np.cos(q[0]+q[1]+q[2])),
        ]) + [x0,y0]
        

model = nengo.Network()
with model:
    # Example 3: a two-joint arm    
    def arm_function(t, angles):
        x0 = 0
        y0 = -125
        
        t = angles[3:]
        q = angles[:3]
        
        
        x1 = x0 + l1*np.sin(q[0])
        y1 = y0 - l1*np.cos(q[0])

        x2 = x1 + l2*np.sin(q[0] + q[1])
        y2 = y1 - l2*np.cos(q[0] + q[1])

        x3 = x2 + l34*np.sin(q[0] + q[1] + q[2])
        y3 = y2 - l34*np.cos(q[0] + q[1] + q[2])

        xt = t[0]
        yt = t[1]

        arm_function._nengo_html_ = '''
        <svg width="100%" height="100%" viewbox="-250 -1000 1000 1000">
            <circle cx="{xt}" cy="{yt}" r="40" style="fill:red"/>
            <line x1="{x0}" y1="{y0}" x2="{x1}" y2="{y1}" style="stroke:black;stroke-width:10px"/>
            <line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" style="stroke:black;stroke-width:10px"/>
            <line x1="{x2}" y1="{y2}" x2="{x3}" y2="{y3}" style="stroke:black;stroke-width:10px"/>
        </svg>
        '''.format(**locals())
    #stim_angles = nengo.Node([0,0,0])
    arm = nengo.Node(arm_function, size_in=5)

    angles = nengo.Node((lambda t, x: x), size_in=3)
    
    stim_target = nengo.Node([0,-578])
    error = nengo.Node(None, size_in=2)
    
    nengo.Connection(stim_target, error)
    nengo.Connection(angles, error, function=fwd, transform=-1)
    
    nengo.Connection(angles, arm[:3])
    nengo.Connection(stim_target, arm[3:])
    
    def Jtrans(t, x):
        q = x[2:]
        dx = x[:2]
        
        J = jacobian(q)
        delta = np.dot(J.T, dx)
        return delta*0.00001
        
    jtrans = nengo.Node(Jtrans, size_in=5)
    
    nengo.Connection(error, jtrans[:2])
    nengo.Connection(angles, jtrans[2:])
    nengo.Connection(jtrans, angles)
    nengo.Connection(angles, angles, synapse=0.1)
    
    
