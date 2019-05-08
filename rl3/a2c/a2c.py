# https://deeplearningcourses.com/c/cutting-edge-artificial-intelligence
import time
import joblib
import numpy as np
import tensorflow as tf
import os


def set_global_seeds(i):
    tf.set_random_seed(i)
    np.random.seed(i)


def cat_entropy(logits):
    a0 = logits - tf.reduce_max(logits, 1, keepdims=True)
    ea0 = tf.exp(a0)
    z0 = tf.reduce_sum(ea0, 1, keepdims=True)
    p0 = ea0 / z0
    return tf.reduce_sum(p0 * (tf.log(z0) - a0), 1)


def find_trainable_variables(key):
    with tf.variable_scope(key):
        return tf.trainable_variables()


def discount_with_dones(rewards, dones, gamma):
    discounted = []
    r = 0
    for reward, done in zip(rewards[::-1], dones[::-1]):
        r = reward + gamma * r * (1. - done)  # fixed off by one bug
        discounted.append(r)
    return discounted[::-1]



class Agent:
    def __init__(self, Network, ob_space, ac_space, nenvs, nsteps, nstack,
                 ent_coef=0.01, vf_coef=0.5, max_grad_norm=0.5, lr=7e-4,
                 alpha=0.99, epsilon=1e-5, total_timesteps=int(80e6)):
        config = tf.ConfigProto(intra_op_parallelism_threads=nenvs,
                                inter_op_parallelism_threads=nenvs)
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        nbatch = nenvs * nsteps

        A = tf.placeholder(tf.int32, [nbatch])
        ADV = tf.placeholder(tf.float32, [nbatch])
        R = tf.placeholder(tf.float32, [nbatch])
        LR = tf.placeholder(tf.float32, [])

        step_model = Network(sess, ob_space, ac_space, nenvs, 1, nstack, reuse=False)
        train_model = Network(sess, ob_space, ac_space, nenvs, nsteps, nstack, reuse=True)

        neglogpac = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=train_model.pi, labels=A)
        pg_loss = tf.reduce_mean(ADV * neglogpac)
        vf_loss = tf.reduce_mean(tf.squared_difference(tf.squeeze(train_model.vf), R) / 2.0)
        entropy = tf.reduce_mean(cat_entropy(train_model.pi))
        loss = pg_loss - entropy * ent_coef + vf_loss * vf_coef

        params = find_trainable_variables("model")
        grads = tf.gradients(loss, params)
        if max_grad_norm is not None:
            grads, grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)
        grads_and_params = list(zip(grads, params))
        trainer = tf.train.RMSPropOptimizer(learning_rate=LR, decay=alpha, epsilon=epsilon)
        _train = trainer.apply_gradients(grads_and_params)

        def train(states, rewards, actions, values):
            advs = rewards - values
            feed_dict = {train_model.X: states, A: actions, ADV: advs, R: rewards, LR: lr}
            policy_loss, value_loss, policy_entropy, _ = sess.run(
                [pg_loss, vf_loss, entropy, _train],
                feed_dict
            )
            return policy_loss, value_loss, policy_entropy

        def save(save_path):
            ps = sess.run(params)
            joblib.dump(ps, save_path)

        def load(load_path):
            loaded_params = joblib.load(load_path)
            restores = []
            for p, loaded_p in zip(params, loaded_params):
                restores.append(p.assign(loaded_p))
            ps = sess.run(restores)

        self.train = train
        self.train_model = train_model
        self.step_model = step_model
        self.step = step_model.step
        self.value = step_model.value
        self.save = save
        self.load = load
        tf.global_variables_initializer().run(session=sess)


class Runner:
    def __init__(self, env, agent, nsteps=5, nstack=4, gamma=0.99):
        self.env = env
        self.agent = agent
        nh, nw, nc = env.observation_space.shape
        nenv = env.num_envs
        self.batch_ob_shape = (nenv * nsteps, nh, nw, nc * nstack)
        self.state = np.zeros((nenv, nh, nw, nc * nstack), dtype=np.uint8)
        self.nc = nc
        obs = env.reset()
        self.update_state(obs)
        self.gamma = gamma
        self.nsteps = nsteps
        self.dones = [False for _ in range(nenv)]
        self.total_rewards = [] # store all workers' total rewards
        self.real_total_rewards = []

    def update_state(self, obs):
        # Do frame-stacking here instead of the FrameStack wrapper to reduce IPC overhead
        self.state = np.roll(self.state, shift=-self.nc, axis=3)
        self.state[:, :, :, -self.nc:] = obs

    def run(self):
        mb_states, mb_rewards, mb_actions, mb_values, mb_dones = [], [], [], [], []
        for n in range(self.nsteps):
            actions, values = self.agent.step(self.state)
            mb_states.append(np.copy(self.state))
            mb_actions.append(actions)
            mb_values.append(values)
            mb_dones.append(self.dones)
            obs, rewards, dones, infos = self.env.step(actions)
            for done, info in zip(dones, infos):
                if done:
                    self.total_rewards.append(info['reward'])
                    if info['total_reward'] != -1:
                        self.real_total_rewards.append(info['total_reward'])
            self.dones = dones
            for n, done in enumerate(dones):
                if done:
                    self.state[n] = self.state[n] * 0
            self.update_state(obs)
            mb_rewards.append(rewards)
        mb_dones.append(self.dones)
        # batch of steps to batch of rollouts
        mb_states = np.asarray(mb_states, dtype=np.uint8).swapaxes(1, 0).reshape(self.batch_ob_shape)
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32).swapaxes(1, 0)
        mb_actions = np.asarray(mb_actions, dtype=np.int32).swapaxes(1, 0)
        mb_values = np.asarray(mb_values, dtype=np.float32).swapaxes(1, 0)
        mb_dones = np.asarray(mb_dones, dtype=np.bool).swapaxes(1, 0)
        mb_dones = mb_dones[:, 1:]
        last_values = self.agent.value(self.state).tolist()
        # discount/bootstrap off value fn
        for n, (rewards, dones, value) in enumerate(zip(mb_rewards, mb_dones, last_values)):
            rewards = rewards.tolist()
            dones = dones.tolist()
            if dones[-1] == 0:
                rewards = discount_with_dones(rewards + [value], dones + [0], self.gamma)[:-1]
            else:
                rewards = discount_with_dones(rewards, dones, self.gamma)
            mb_rewards[n] = rewards
        mb_rewards = mb_rewards.flatten()
        mb_actions = mb_actions.flatten()
        mb_values = mb_values.flatten()
        return mb_states, mb_rewards, mb_actions, mb_values


def learn(network, env, seed, new_session=True,  nsteps=5, nstack=4, total_timesteps=int(80e6),
          vf_coef=0.5, ent_coef=0.01, max_grad_norm=0.5, lr=7e-4,
          epsilon=1e-5, alpha=0.99, gamma=0.99, log_interval=1000):
    tf.reset_default_graph()
    set_global_seeds(seed)

    nenvs = env.num_envs
    env_id = env.env_id
    save_name = os.path.join('models', env_id + '.save')
    ob_space = env.observation_space
    ac_space = env.action_space
    agent = Agent(Network=network, ob_space=ob_space, ac_space=ac_space, nenvs=nenvs,
                  nsteps=nsteps, nstack=nstack,
                  ent_coef=ent_coef, vf_coef=vf_coef,
                  max_grad_norm=max_grad_norm,
                  lr=lr, alpha=alpha, epsilon=epsilon, total_timesteps=total_timesteps)
    if os.path.exists(save_name):
        agent.load(save_name)

    runner = Runner(env, agent, nsteps=nsteps, nstack=nstack, gamma=gamma)

    nbatch = nenvs * nsteps
    tstart = time.time()
    for update in range(1, total_timesteps // nbatch + 1):
        states, rewards, actions, values = runner.run()
        policy_loss, value_loss, policy_entropy = agent.train(
            states, rewards, actions, values)
        nseconds = time.time() - tstart
        fps = int((update * nbatch) / nseconds)
        if update % log_interval == 0 or update == 1:
            print(' - - - - - - - ')
            print("nupdates", update)
            print("total_timesteps", update * nbatch)
            print("fps", fps)
            print("policy_entropy", float(policy_entropy))
            print("value_loss", float(value_loss))

            # total reward
            r = runner.total_rewards[-100:] # get last 100
            tr = runner.real_total_rewards[-100:]
            if len(r) == 100:
                print("avg reward (last 100):", np.mean(r))
            if len(tr) == 100:
                print("avg total reward (last 100):", np.mean(tr))
                print("max (last 100):", np.max(tr))

            agent.save(save_name)

    env.close()
    agent.save(save_name)
