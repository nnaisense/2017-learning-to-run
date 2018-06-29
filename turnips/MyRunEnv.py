import opensim
import math
import numpy as np
import os
from itertools import chain
from osim.env import OsimEnv
from osim.env import RunEnv
import osim.env
from multiprocessing import Process, Pipe
from gym.spaces import Box
from turnips.MyRunEnvLogger import MyRunEnvLogger


def flatten(listOfLists):
    "Flatten one level of nesting"
    return chain.from_iterable(listOfLists)


class MyRunEnv(OsimEnv):
    STATE_PELVIS_X = 1
    STATE_PELVIS_Y = 2
    MUSCLES_PSOAS_R = 3
    MUSCLES_PSOAS_L = 11

    num_obstacles = 0
    max_obstacles = None

    model_path = os.path.join(osim.env.__path__[0], '../models/gait9dof18musc.osim')
    ligamentSet = []
    verbose = False
    pelvis = None
    env_desc = {"obstacles": [], "muscles": [1] * 18}

    ninput = 41
    noutput = 18

    def __init__(self, visualize=True, max_obstacles=3):
        self.max_obstacles = max_obstacles
        super(MyRunEnv, self).__init__(visualize=False, noutput=self.noutput)
        self.osim_model.model.setUseVisualizer(visualize)
        self.create_obstacles()
        state = self.osim_model.model.initSystem()

        if visualize:
            manager = opensim.Manager(self.osim_model.model)
            manager.setInitialTime(-0.00001)
            manager.setFinalTime(0.0)
            manager.integrate(state)

    def setup(self, difficulty, seed=None):
        # create the new env
        # set up obstacles
        self.env_desc = self.generate_env(difficulty, seed, self.max_obstacles)

        self.clear_obstacles(self.osim_model.state)
        for x, y, r in self.env_desc['obstacles']:
            self.add_obstacle(self.osim_model.state, x, y, r)

        # set up muscle strength
        self.osim_model.set_strength(self.env_desc['muscles'])

    def reset(self, difficulty=2, seed=None):
        super(MyRunEnv, self).reset()
        self.istep = 0
        self.setup(difficulty, seed)
        self.last_state = self.get_observation()
        self.current_state = self.last_state
        return self.last_state

    def compute_reward(self):
        # Compute ligaments penalty
        lig_pen = 0
        # Get ligaments
        for j in range(20, 26):
            lig = opensim.CoordinateLimitForce.safeDownCast(self.osim_model.forceSet.get(j))
            lig_pen += lig.calcLimitForce(self.osim_model.state) ** 2

        # Get the pelvis X delta
        delta_x = self.current_state[self.STATE_PELVIS_X] - self.last_state[self.STATE_PELVIS_X]

        self.ligament_reward = -math.sqrt(lig_pen) * 10e-8
        self.delta_x_reward = delta_x
        return self.delta_x_reward + self.ligament_reward

    def is_pelvis_too_low(self):
        return (self.current_state[self.STATE_PELVIS_Y] < 0.65)

    def is_done(self):
        return self.is_pelvis_too_low() or (self.istep >= self.spec.timestep_limit)

    def configure(self):
        super(MyRunEnv, self).configure()

        if self.verbose:
            print("JOINTS")
            for i in range(11):
                print(i, self.osim_model.jointSet.get(i).getName())
            print("\nBODIES")
            for i in range(13):
                print(i, self.osim_model.bodySet.get(i).getName())
            print("\nMUSCLES")
            for i in range(18):
                print(i, self.osim_model.muscleSet.get(i).getName())
            print("\nFORCES")
            for i in range(26):
                print(i, self.osim_model.forceSet.get(i).getName())
            print("")

        # for i in range(18):
        #     m = opensim.Thelen2003Muscle.safeDownCast(self.osim_model.muscleSet.get(i))
        #     m.setActivationTimeConstant(0.0001) # default 0.01
        #     m.setDeactivationTimeConstant(0.0001) # default 0.04

        # The only joint that has to be cast
        self.pelvis = opensim.PlanarJoint.safeDownCast(self.osim_model.get_joint("ground_pelvis"))

    def next_obstacle(self):
        obstacles = self.env_desc['obstacles']
        x = self.pelvis.getCoordinate(self.STATE_PELVIS_X).getValue(self.osim_model.state)
        for obstacle in obstacles:
            if obstacle[0] + obstacle[2] < x:
                continue
            else:
                ret = list(obstacle)
                ret[0] = ret[0] - x
                return ret
        return [100, 0, 0]

    def _step(self, action):
        self.last_state = self.current_state
        res = super(MyRunEnv, self)._step(action)
        # Add additional information to info
        info = res[-1]
        info['delta_x_reward'] = self.delta_x_reward
        info['ligament_reward'] = self.ligament_reward
        return res

    def get_observation(self):
        bodies = ['head', 'pelvis', 'torso', 'toes_l', 'toes_r', 'talus_l', 'talus_r']

        pelvis_pos = [self.pelvis.getCoordinate(i).getValue(self.osim_model.state) for i in range(3)]
        pelvis_vel = [self.pelvis.getCoordinate(i).getSpeedValue(self.osim_model.state) for i in range(3)]

        jnts = ['hip_r', 'knee_r', 'ankle_r', 'hip_l', 'knee_l', 'ankle_l']
        joint_angles = [self.osim_model.get_joint(jnts[i]).getCoordinate().getValue(self.osim_model.state) for i in
                        range(6)]
        joint_vel = [self.osim_model.get_joint(jnts[i]).getCoordinate().getSpeedValue(self.osim_model.state) for i in
                     range(6)]

        mass_pos = [self.osim_model.model.calcMassCenterPosition(self.osim_model.state)[i] for i in range(2)]
        mass_vel = [self.osim_model.model.calcMassCenterVelocity(self.osim_model.state)[i] for i in range(2)]

        body_transforms = [
            [self.osim_model.get_body(body).getTransformInGround(self.osim_model.state).p()[i] for i in range(2)] for
            body in bodies]

        muscles = [self.env_desc['muscles'][self.MUSCLES_PSOAS_L], self.env_desc['muscles'][self.MUSCLES_PSOAS_R]]

        # see the next obstacle
        obstacle = self.next_obstacle()

        #        feet = [opensim.HuntCrossleyForce.safeDownCast(self.osim_model.forceSet.get(j)) for j in range(20,22)]
        self.current_state = pelvis_pos + pelvis_vel + joint_angles + joint_vel + mass_pos + mass_vel + list(
            flatten(body_transforms)) + muscles + obstacle

        self.current_muscle_activations = [self.osim_model.model.getMuscles().get(i).getActivation(self.osim_model.state) for i in range(18)]

        return self.current_state

    def create_obstacles(self):
        x = 0
        y = 0
        r = 0.1
        for i in range(self.max_obstacles):
            name = i.__str__()
            blockos = opensim.Body(name + '-block', 0.0001, opensim.Vec3(0), opensim.Inertia(1, 1, .0001, 0, 0, 0));
            pj = opensim.PlanarJoint(name + '-joint',
                                     self.osim_model.model.getGround(),  # PhysicalFrame
                                     opensim.Vec3(0, 0, 0),
                                     opensim.Vec3(0, 0, 0),
                                     blockos,  # PhysicalFrame
                                     opensim.Vec3(0, 0, 0),
                                     opensim.Vec3(0, 0, 0))

            self.osim_model.model.addJoint(pj)
            self.osim_model.model.addBody(blockos)

            block = opensim.ContactSphere(r, opensim.Vec3(0, 0, 0), blockos)
            block.setName(name + '-contact')
            self.osim_model.model.addContactGeometry(block)

            force = opensim.HuntCrossleyForce()
            force.setName(name + '-force')

            force.addGeometry(name + '-contact')
            force.addGeometry("r_heel")
            force.addGeometry("l_heel")
            force.addGeometry("r_toe")
            force.addGeometry("l_toe")

            force.setStiffness(1.0e6 / r)
            force.setDissipation(1e-5)
            force.setStaticFriction(0.0)
            force.setDynamicFriction(0.0)
            force.setViscousFriction(0.0)

            self.osim_model.model.addForce(force);

    def clear_obstacles(self, state):
        for j in range(0, self.max_obstacles):
            joint_generic = self.osim_model.get_joint("%d-joint" % j)
            joint = opensim.PlanarJoint.safeDownCast(joint_generic)
            joint.getCoordinate(1).setValue(state, 0)
            joint.getCoordinate(2).setValue(state, -0.1)

            contact_generic = self.osim_model.get_contact_geometry("%d-contact" % j)
            contact = opensim.ContactSphere.safeDownCast(contact_generic)
            contact.setRadius(0.0001)

            for i in range(3):
                joint.getCoordinate(i).setLocked(state, True)

        self.num_obstacles = 0
        pass

    def add_obstacle(self, state, x, y, r):
        # set obstacle number num_obstacles
        contact_generic = self.osim_model.get_contact_geometry("%d-contact" % self.num_obstacles)
        contact = opensim.ContactSphere.safeDownCast(contact_generic)
        contact.setRadius(r)

        force_generic = self.osim_model.get_force("%d-force" % self.num_obstacles)
        force = opensim.HuntCrossleyForce.safeDownCast(force_generic)
        force.setStiffness(1.0e6 / r)

        joint_generic = self.osim_model.get_joint("%d-joint" % self.num_obstacles)
        joint = opensim.PlanarJoint.safeDownCast(joint_generic)

        newpos = [x, y]
        for i in range(2):
            joint.getCoordinate(1 + i).setLocked(state, False)
            joint.getCoordinate(1 + i).setValue(state, newpos[i], False)
            joint.getCoordinate(1 + i).setLocked(state, True)

        self.num_obstacles += 1
        pass

    def generate_env(self, difficulty, seed, max_obstacles):
        assert difficulty <= 2, "Currently I don't support diff=3 any longer"
        if seed is not None:
            np.random.seed(seed)  # seed the RNG if seed is provided

        # obstacles
        num_obstacles = 0
        xs = []
        ys = []
        rs = []

        if 0 < difficulty:
            num_obstacles = min(3, max_obstacles)
            xs = np.random.uniform(1.0, 5.0, num_obstacles)
            ys = np.random.uniform(-0.25, 0.25, num_obstacles)
            rs = [0.05 + r for r in np.random.exponential(0.05, num_obstacles)]

        if 0 < difficulty and 3 < max_obstacles:
            extra_obstacles = max(min(20, max_obstacles) - num_obstacles, 0)
            xs = np.concatenate([xs, (np.cumsum(np.random.uniform(2.0, 4.0, extra_obstacles)) + 5)])
            ys = np.concatenate([ys, np.random.uniform(-0.05, 0.25, extra_obstacles)])
            rs = rs + [0.05 + r for r in np.random.exponential(0.05, extra_obstacles)]
            num_obstacles = len(xs)

        ys = map(lambda xy: xy[0] * xy[1], list(zip(ys, rs)))

        # muscle strength
        rpsoas = 1
        lpsoas = 1
        if difficulty >= 2:
            rpsoas = 1 - np.random.normal(0, 0.1)
            lpsoas = 1 - np.random.normal(0, 0.1)
            rpsoas = max(0.5, rpsoas)
            lpsoas = max(0.5, lpsoas)

        muscles = [1] * 18

        # modify only psoas
        muscles[self.MUSCLES_PSOAS_R] = rpsoas
        muscles[self.MUSCLES_PSOAS_L] = lpsoas

        obstacles = list(zip(xs, ys, rs))
        obstacles.sort()

        return {
            'muscles': muscles,
            'obstacles': obstacles
        }

# bind our custom version of pelvis too low judgement function to original env
def bind_alternative_pelvis_judgement(runenv, val):
    def is_pelvis_too_low(self):
        return (self.current_state[self.STATE_PELVIS_Y] < val)
    import types
    runenv.is_pelvis_too_low = types.MethodType(is_pelvis_too_low, runenv)


# separate process that holds a separate RunEnv instance.
# This has to be done since RunEnv() in the same process result in interleaved running of simulations.
def standalone_headless_isolated(conn, visualize, n_obstacles, run_logs_dir, additional_info, higher_pelvis=0.65):
    try:
        e = RunEnv(visualize=visualize, max_obstacles=n_obstacles)
        if higher_pelvis != 0.65:
            bind_alternative_pelvis_judgement(e, higher_pelvis)
        e = MyRunEnvLogger(e, log_dir=run_logs_dir, additional_info=additional_info)

        while True:
            msg = conn.recv()

            # messages should be tuples,
            # msg[0] should be string

            if msg[0] == 'reset':
                o = e.reset(difficulty=msg[1], seed=msg[2])
                conn.send(o)
            elif msg[0] == 'step':
                ordi = e.step(msg[1])
                conn.send(ordi)
            elif msg[0] == 'close':
                e.close()
                conn.send(None)

                import psutil
                current_process = psutil.Process()
                children = current_process.children(recursive=True)
                for child in children:
                    child.terminate()
                return
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        conn.send(e)


class IsolatedMyRunEnv: # Environment Instance
    metadata = {'render.modes': ['human', 'rgb_array'], 'video.frames_per_second': 10}

    def __init__(self, visualize=True, n_obstacles=3, run_logs_dir=None, additional_info={}, step_timeout=None, higher_pelvis=0.65):
        self.running = False
        self.action_space = Box(low=0, high=1, shape=[18])
        self.observation_space = Box(low=-3, high=+3, shape=[41])
        self.reward_range = [0, 100]
        self.visualize = visualize
        self.n_obstacles = n_obstacles
        self.run_logs_dir = run_logs_dir
        self.additional_info = additional_info
        self.pc = None
        self.step_timeout = step_timeout
        self.lastobs = None
        self.p = None
        self.higher_pelvis = higher_pelvis

    def reset(self, difficulty, seed):
        if self.running:
            self.pc.send(('close',))
            self.p.join()

        self.pc, self.cc = Pipe()
        self.p = Process(
            target=standalone_headless_isolated,
            args=(self.cc, self.visualize, self.n_obstacles, self.run_logs_dir, self.additional_info, self.higher_pelvis),
        )
        # self.p.daemon = True
        self.p.start()
        self.pc.send(('reset', difficulty, seed))
        self.running = True

        res = self.pc.recv()
        self.lastobs = res
        if isinstance(res, Exception):
            raise res
        return res

    def step(self, actions):
        self.pc.send(('step', list(actions)), )
        finished = self.pc.poll(self.step_timeout)
        if finished:
            res = self.pc.recv()
            if isinstance(res, Exception):
                raise res
            self.lastobs = res[0]
            return res
        else:
            print('Env Timeouted!')
            self.p.terminate()
            return self.lastobs, -0.045, True, {"Timeouted": True}

    def close(self):
        if self.pc is not None:
            self.pc.send(('close',))
            res = self.pc.recv()
            if isinstance(res, Exception):
                raise res

    def __del__(self):
        if self.pc is not None:
            self.pc.send(('close',))
        return
