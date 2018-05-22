import tensorflow as tf
import numpy as np


class MonteCarlo(object):
    def __init__(self):
        self.epsilon = 0.6

    def step(self,obs,sess):
        obs = np.expand_dims(obs,0)
        if np.random.rand() < np.max([self.epsilon-tf.train.global_step(sess,self.global_step)*(0.6/300.0),0]):
            action = np.random.choice([1,0])

        else:
            if sess.run(self.value,{self.obs_ph:obs,self.actions_ph:np.expand_dims([0],0)}) > \
                    sess.run(self.value,{self.obs_ph:obs,self.actions_ph:np.expand_dims([1],0)}):
                action = 0
            elif sess.run(self.value,{self.obs_ph:obs,self.actions_ph:np.expand_dims([1],0)}) > \
                    sess.run(self.value,{self.obs_ph:obs,self.actions_ph:np.expand_dims([0],0)}):
                action = 1
            else:
                action = np.random.choice([1, 0])

        return action

    def calcReturns(self,rewardsAll,obsAll,actionsAll,disc,sess):
        return np.expand_dims([np.sum(np.multiply(rewardsAll[j:], [disc ** k for k in range(len(rewardsAll) - j)]))
         for j in range(len(rewardsAll))],1)

    def network(self,x):
        first_layer = tf.layers.dense(inputs=x,units=10,activation=tf.nn.relu)
        return tf.layers.dense(inputs=first_layer,units=1)


    def build(self,sess):
        self.obs_ph = tf.placeholder(tf.float32,shape=[None,4])
        self.actions_ph = tf.placeholder(tf.float32,shape=[None,1])
        self.returns_ph = tf.placeholder(tf.float32,shape=[None,1])
        self.value = self.network(tf.concat([self.obs_ph,self.actions_ph],1))
        self.loss = tf.losses.mean_squared_error(self.returns_ph,self.value)
        optimizer = tf.train.GradientDescentOptimizer(0.05)
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        self.train_op = optimizer.minimize(self.loss,self.global_step)
        init = tf.global_variables_initializer()
        sess.run(init)


    def train(self,feed,sess):
        _,loss_value = sess.run([self.train_op,self.loss],{self.obs_ph:feed[0][:-1],self.actions_ph:feed[1],self.returns_ph:feed[2]})
        print(loss_value)



class Sarsa(object):
    def __init__(self):
        self.epsilon = 0.6

    def step(self,obs,sess):
        obs = np.expand_dims(obs,0)
        if np.random.rand() < np.max([self.epsilon-tf.train.global_step(sess,self.global_step)*(0.6/500.0),0]):
            action = np.random.choice([1,0])

        else:
            if sess.run(self.value,{self.obs_ph:obs,self.actions_ph:np.expand_dims([0],0)}) > \
                    sess.run(self.value,{self.obs_ph:obs,self.actions_ph:np.expand_dims([1],0)}):
                action = 0
            elif sess.run(self.value,{self.obs_ph:obs,self.actions_ph:np.expand_dims([1],0)}) > \
                    sess.run(self.value,{self.obs_ph:obs,self.actions_ph:np.expand_dims([0],0)}):
                action = 1
            else:
                action = np.random.choice([1, 0])
        return action

    def calcReturns(self,rewardsAll,obsAll,actionsAll,disc,sess):
        nextObsAll = obsAll[1:-1]
        value = sess.run(self.value,{self.obs_ph:nextObsAll,self.actions_ph:np.expand_dims(actionsAll[1:],1)})
        returnsAll = np.expand_dims(rewardsAll[:-1],1) + disc*value
        returnsAll = np.expand_dims(np.append(returnsAll,np.expand_dims(rewardsAll[-1],1)),1)
        return returnsAll

    def network(self,x):
        first_layer = tf.layers.dense(inputs=x,units=10,activation=tf.nn.relu)
        return tf.layers.dense(inputs=first_layer,units=1)


    def build(self,sess):
        self.obs_ph = tf.placeholder(tf.float32,shape=[None,4])
        self.actions_ph = tf.placeholder(tf.float32,shape=[None,1])
        self.returns_ph = tf.placeholder(tf.float32,shape=[None,1])
        self.value = self.network(tf.concat([self.obs_ph,self.actions_ph],1))
        self.loss = tf.losses.mean_squared_error(self.returns_ph,self.value)
        optimizer = tf.train.GradientDescentOptimizer(0.05)
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        self.train_op = optimizer.minimize(self.loss,self.global_step)
        init = tf.global_variables_initializer()
        sess.run(init)


    def train(self,feed,sess):
        _,loss_value = sess.run([self.train_op,self.loss],{self.obs_ph:feed[0][:-1],self.actions_ph:feed[1],self.returns_ph:feed[2]})
        print(loss_value)



class Sarsa20(object):
    def __init__(self):
        self.epsilon = 0.6

    def step(self,obs,sess):
        obs = np.expand_dims(obs,0)
        if np.random.rand() < np.max([self.epsilon-tf.train.global_step(sess,self.global_step)*(0.6/500.0),0]):
            action = np.random.choice([1,0])

        else:
            if sess.run(self.value,{self.obs_ph:obs,self.actions_ph:np.expand_dims([0],0)}) > \
                    sess.run(self.value,{self.obs_ph:obs,self.actions_ph:np.expand_dims([1],0)}):
                action = 0
            elif sess.run(self.value,{self.obs_ph:obs,self.actions_ph:np.expand_dims([1],0)}) > \
                    sess.run(self.value,{self.obs_ph:obs,self.actions_ph:np.expand_dims([0],0)}):
                action = 1
            else:
                action = np.random.choice([1, 0])
        return action

    def calcReturns(self,rewardsAll,obsAll,actionsAll,disc,sess):
        returnsAll = np.empty(len(rewardsAll))
        for i in range(len(rewardsAll)):
            if (len(rewardsAll) - i < 22):
                returnsAll[i] = np.sum(np.multiply(rewardsAll[i:], [disc ** k for k in range(len(rewardsAll) - i)]))
            else:
                value = sess.run(self.value,{self.obs_ph: np.expand_dims(obsAll[i+21],0),
                                  self.actions_ph: np.expand_dims([actionsAll[i+21]], 0)})
                returnsAll[i] = np.sum(np.multiply(rewardsAll[i:i+21], [disc ** k for k in range(21)])) + disc**21 * value

        returnsAll = np.expand_dims(returnsAll,1)
        return returnsAll

    def network(self,x):
        first_layer = tf.layers.dense(inputs=x,units=10,activation=tf.nn.relu)
        return tf.layers.dense(inputs=first_layer,units=1)


    def build(self,sess):
        self.obs_ph = tf.placeholder(tf.float32,shape=[None,4])
        self.actions_ph = tf.placeholder(tf.float32,shape=[None,1])
        self.returns_ph = tf.placeholder(tf.float32,shape=[None,1])
        self.value = self.network(tf.concat([self.obs_ph,self.actions_ph],1))
        self.loss = tf.losses.mean_squared_error(self.returns_ph,self.value)
        optimizer = tf.train.GradientDescentOptimizer(0.05)
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        self.train_op = optimizer.minimize(self.loss,self.global_step)
        init = tf.global_variables_initializer()
        sess.run(init)


    def train(self,feed,sess):
        _,loss_value = sess.run([self.train_op,self.loss],{self.obs_ph:feed[0][:-1],self.actions_ph:feed[1],self.returns_ph:feed[2]})
        print(loss_value)


