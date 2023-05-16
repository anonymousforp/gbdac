"""Implementation of QLearning Algorithms."""
import numpy as np
import torch, time
import torch.nn.functional as F
from gym.wrappers import ClipAction
from oraaclib.dataset import Observation
from oraaclib.util.losses import quantile_huber_loss
from oraaclib.util.utilities import Wang_distortion, CPW, Power
from torch.distributions import uniform
from torch.distributions.normal import Normal
from .abstract_agent import AbstractAgent

__all__ = ['ORAAC']


class ORAAC(AbstractAgent):
    """
    """

    def __init__(self, env, policy, critic, target_policy,
                 target_critic,
                 hyper_params, dataset,
                 tb=None, vae=None, eval=False,
                 logger=None, save_model=False, name_save=None,
                 early_stopper_rew=None, early_stopper_var=None,
                 render=False):

        super().__init__()

        self.policy = policy
        self.vae = vae
        self.env = env
        self.tb = tb
        self.eval = eval
        self.logger = logger
        self.render = render
        if self.policy.__class__.__name__ == 'RAAC_Actor':
            self.with_IL = False
            self.train_vae = False
        else:
            self.with_IL = True # True if ORAAC
            self.train_vae = False
            self.train_vae_diff = True
        if not self.eval:
            self.critic = critic
            self.target_critic = target_critic
            self.target_policy = target_policy

            self.gamma = hyper_params['gamma']
            self.policy_update_freq = hyper_params['target_update_freq']
            self.batch_size = hyper_params['batch_size']
            self.K = hyper_params['n_quantiles_policy']
            self.N = hyper_params['n_quantiles_critic']

            if hyper_params['risk_distortion'] == 'cvar':
                self.alpha_cvar = hyper_params['alpha_cvar']
                self.distr_taus_risk = uniform.Uniform(0., self.alpha_cvar)
            elif hyper_params['risk_distortion'] == 'wang':
                self.distr_taus_risk = Wang_distortion()
            elif hyper_params['risk_distortion'] == 'cpw':
                self.distr_taus_risk = CPW()
            elif hyper_params['risk_distortion'] == 'power':
                self.distr_taus_risk = Power()
            else:
                raise ValueError("No distortion found")

            self.distr_taus_uniform = uniform.Uniform(0., 1.)
            self.ActionClipper = ClipAction(self.env)

            self.optimizer_actor = torch.optim.Adam(
                params=self.policy.parameters(),
                lr=hyper_params['lr_actor'])

            self.optimizer_critic = torch.optim.Adam(
                params=self.critic.parameters(),
                lr=hyper_params['lr_critic'])
            if self.train_vae:
                self.optimizer_vae = torch.optim.Adam(self.vae.parameters())
            # self.optimizer_vae_diffusion = torch.optim.Adam(self.vae.parameters(), lr=2e4)
            self.dataset = dataset

            self.name_save = name_save
            self.bool_save_model = save_model
            self.early_stopper_var = early_stopper_var
            self.early_stopper_rew = early_stopper_rew
            self.report_file = self.name_save+'_total_evol.json' \
                if self.name_save else None

            # Timers
            self.t_critic = self.t_actor = self.t_td = self.t_update = \
                self.t_vae = self.t_vae_loss = \
                self.t_vae_back = self.counter_vae_loss = 0
            self.vae_loss_vector = []

    def train(self):
        super().train_step()  # count number of training steps

        obs = self.dataset.get_batch(batch_size=self.batch_size)
        state, action, reward, done, next_state = obs
        # Diffusion Training
        dataset = {'observations': state.detach().numpy(), 'actions': action.detach().numpy(),
                   'next_observations': next_state.detach().numpy(), 'rewards': reward.detach().numpy(),
                   'terminals': done.detach().numpy()}
        data_sampler = Data_Sampler(dataset, device='cpu', reward_tune='no')
        train_iter = 0

        if self.train_vae_diff:
            while train_iter < 2000:
                loss_metric = self.vae.train(data_sampler, iterations=1000, batch_size=128, log_writer=None)
                train_iter += 2000

                curr_epoch = int(train_iter // 1000)

                # Logging
                print(f"Train step: {train_iter}")
                print('Trained Epochs', curr_epoch)
                print('BC Loss', np.mean(loss_metric['bc_loss']))
                # print('QL Loss', np.mean(loss_metric['ql_loss']))
                # print('Actor Loss', np.mean(loss_metric['actor_loss']))
                # print('Critic Loss', np.mean(loss_metric['critic_loss']))
                self.train_vae_diff = False if np.mean(loss_metric['bc_loss']) < 0.11 else True

        # Variational Auto - Encoder Training:
        if self.train_vae:
            recon, mean, std = self.vae(state.clone(), action.clone())
            # mean and std size = [batch_size x latent_dim]
            recon_loss = F.mse_loss(recon, action)
            KL_loss = self.compute_KL_loss(
                mean, std).mean()  # potentially quicker
            vae_loss = recon_loss + 0.5 * KL_loss

            # Early Stopper VAE
            if self.num_train_steps > 10000 and not self.num_train_steps % 100:
                self.vae_loss_vector.append(vae_loss.data.numpy())
                if len(self.vae_loss_vector) > 50:
                    if np.abs(
                            max(self.vae_loss_vector[-50:]) -
                            min(self.vae_loss_vector[-50:])) <= 0.03:
                        self.counter_vae_loss += 1
                    else:
                        self.counter_vae_loss = 0
                if self.counter_vae_loss >= 10:
                    self.train_vae = False
                    self.vae_loss_vector = []

            self.optimizer_vae.zero_grad()
            vae_loss.backward()
            self.optimizer_vae.step()

        # Critic Training:
        current_Z, target_Z, tau_k = self.td(state, action, reward, next_state, done)

        # update critic network
        self.critic_loss = quantile_huber_loss(target_Z, current_Z, tau_k)
        self.optimizer_critic.zero_grad()
        self.critic_loss.backward()
        self.optimizer_critic.step()

        # Actor Training:
        if self.num_train_steps % self.policy_update_freq == 0:

            self.actor_loss = -self.compute_actor_loss(state)
            self.optimizer_actor.zero_grad()
            self.actor_loss.backward()
            self.optimizer_actor.step()

            # Update target networks (softly)
            # call @params.setter in IQN NN
            self.target_critic.params = self.critic.params
            # call @params.setter in ORAAC NN
            self.target_policy.params = self.policy.params
        if self.tb is not None:
            if self.num_train_steps % self.policy_update_freq == 0 and \
                    self.num_train_steps % 100 == 0:
                self.tb.add_scalar('Loss/actor_loss',
                                   self.actor_loss, self.num_train_steps)
                self.tb.add_scalar('Loss/critic_loss',
                                   self.critic_loss, self.num_train_steps)
                if self.train_vae:
                    self.tb.add_scalar('Loss/vae_loss',
                                       vae_loss, self.num_train_steps)

    def td(self, state, action, reward, next_state, done):
        tau_k = self.distr_taus_uniform.sample(
            (self.N,))  # [batch_size x N]
        tau_k_ = self.distr_taus_uniform.sample(
            (self.N,))  # [batch_size x N]

        # [batch_size x num_confidences]
        Z_tau_K = self.critic.get_sampled_Z(state, tau_k, action)
        with torch.no_grad():
            next_action = self.compute_next_action(next_state)
            Z_next_tau_K = self.target_critic.get_sampled_Z(
                next_state, tau_k_, next_action)

            done = done.unsqueeze(-1).expand_as(Z_next_tau_K)
            reward = reward.unsqueeze(-1).expand_as(Z_next_tau_K)
            target_Z_tau_K = reward + self.gamma * Z_next_tau_K * (1 - done)
        return Z_tau_K, target_Z_tau_K, tau_k

    def compute_next_action(self, next_state):
        with torch.no_grad():
            if self.with_IL:  # ORAAC
                # Sample action from state-conditioned marginal likelihood
                # learnt by VAE
                if self.train_vae:
                    vae_action = self.vae.decode(next_state)
                else:
                    vae_action = self.vae.sample_action(next_state)

                # Final action: vae_action + lamda * risk_averse action

                next_action = self.target_policy(
                    next_state, vae_action)
            else:  # RAAC, no VAE
                next_action = self.target_policy(
                    next_state)
        return next_action

    def compute_KL_loss(self, mu, std):
        u = Normal(loc=torch.zeros_like(mu), scale=torch.ones_like(mu))
        v = Normal(loc=mu, scale=std)
        loss = torch.distributions.kl_divergence(v, u)
        return loss

    def compute_actor_loss(self, state):
        """ Compute CVaR of the reward distribution given state and action
        selected by risk-averse policy
        """
        if self.with_IL:
            # Sample action from state-conditioned marginal likelihood
            # learnt by VAE
            if self.train_vae:
                vae_action = self.vae.decode(state).detach()
            else:
                vae_action = self.vae.sample_action(state)

            action = self.policy(state, vae_action)
        else:
            action = self.policy(state)
        tau_actor_k = self.distr_taus_risk.sample((self.K,))
        tail_samples = self.critic.get_sampled_Z(
            state, tau_actor_k, action)  # [batch_size x K]

        cvar = tail_samples.mean()
        return cvar

    def act(self, state):  # only for evaluation
        with torch.no_grad():
            state = torch.FloatTensor(state.reshape(1, -1))
            if self.with_IL:  # ORAAC
                # latent vector to zeros if in evaluation mode
                if self.train_vae:
                    vae_action = self.vae.decode(state).detach()
                else:
                    vae_action = self.vae.sample_action(state).detach()
                # vae_action = self.vae.decode(state, eval=self.eval)
                action = self.policy(
                    state, vae_action)
            else:  # RAAC
                action = self.policy(state)

        return action.data.numpy()

    def evaluate_model(self, max_episode_steps, times_eval=1):
        self.times_eval = times_eval
        with torch.no_grad():
            for i in range(times_eval):
                super().start_episode_offline(eval=self.eval)
                self.eval_episode = int(self.num_eval_episodes/times_eval)
                state = self.env.reset()
                done = False
                c_reward = 0
                while not done:
                    if self.render:
                        self.env.render()
                    action = self.act(state)

                    action = action.reshape((6))
                    next_state, reward, done, info = self.env.step(action)
                    observation = Observation(state=state,
                                              action=action,
                                              reward=reward,
                                              next_state=next_state,
                                              done=done)
                    self.observe_offline(observation, info, eval=self.eval)
                    state = next_state
                    if max_episode_steps <= self.episodes_eval_steps[-1]:
                        break
                super().end_episode_offline()
                # print(self.mean_, self.cvar_)
                print(f'Num steps {self.episodes_eval_steps[-1]}/'
                      f'{max_episode_steps}')
                print(f'Fraction Risky times:'
                      f'{self.fraction_risky_times(self.times_eval):.2f}\n\n')

        if self.mean_ >= 400:
            self.save_final_model()
        if self.logger:
            print("logging")
            self.log_data()
            self.report_performance()
        if self.tb is not None:
            self.tb_data()
        if not self.eval_episode % 100:
            self.report_performance()

    def log_data(self):
        self.logger.add(
            **{"eval_mean_reward": self.mean_,
                "eval_cvar_reward": self.cvar_,
                "mean_ep_steps": self.mean_ep_steps(self.times_eval),
                "mean_vel_episodes": self.mean_vel_episodes(self.times_eval),
                "mean_risky_times": self.mean_risky_times(self.times_eval),
                "fraction_risky_times":
                    self.fraction_risky_times(self.times_eval)
               })
        if self.eval:
            self.logger.add(**{"angles": self.logs['episodes_angles']})
            self.logger.add(**{"velocities": self.logs['episodes_vels']})
        # Save performance results during evaluation
        if not self.num_train_steps % 1000 or self.eval:
            self.logger.export_to_json()

    def tb_data(self):

        self.tb.add_scalar('Risk/Mean_risky_times',
                           self.mean_risky_times(self.times_eval),
                           self.eval_episode)
        self.tb.add_scalar('Risk/Fraction_risky_times',
                           self.fraction_risky_times(self.times_eval),
                           self.eval_episode)
        self.tb.add_scalar('Risk/Mean_vel_episodes',
                           self.mean_vel_episodes(self.times_eval),
                           self.eval_episode)
        self.tb.add_scalar('Performance/Mean_eval_steps',
                           self.mean_ep_steps(self.times_eval),
                           self.eval_episode)
        self.tb.add_scalar('Performance/CVAR eval',
                           self.cvar_,
                           self.eval_episode)
        self.tb.add_scalar('Performance/Mean eval',
                           self.mean_,
                           self.eval_episode)

    def report_performance(self):
        dashes = '-'*30+'\n'
        tex = f'Evaluation episode: {self.eval_episode}\n\n'\
            f'Mean Cumulative reward: {self.mean_:.2f}\n'\
            f'CVaR Cumulative reward: {self.cvar_:.2f}\n'\
            f'Mean Episode Steps {self.mean_ep_steps(self.times_eval)}\n'\
            f'Fraction Risky times:'\
            f'{self.fraction_risky_times(self.times_eval):.2f}\n\n'
        if not self.eval:
            tex_train = 'Best performance seen so far:\n'\
                f'  Best mean: {self.early_stopper_rew.best_mean:.2f}.'\
                f'  Patience {self.early_stopper_rew.counter}\n'\
                f'  Best cvar: {self.early_stopper_var.cvar_max:.2f}.'\
                f'  Patience {self.early_stopper_var.counter}\n'
        else:
            tex_train = ''
        print(dashes + tex + tex_train + dashes)

    def save_model(self):
        if not self.bool_save_model:
            pass
        elif self.eval_episode > 30:
            print("start saving.....")
            model_dict = {
                'critic': self.critic.state_dict(),
                'actor': self.policy.state_dict(),
                'vae': self.vae.state_dict()}

            self.early_stopper_rew.call_mean(
                score=self.mean_,
                episode_num=self.eval_episode,
                model_dict=model_dict)
            self.early_stopper_var.call_cvar_mean(
                mean=self.mean_,
                cvar=self.cvar_,
                episode_num=self.eval_episode,
                model_dict=model_dict)

    def save_final_model(self):
        if not self.bool_save_model:
            pass
        else:
            model_dict = {
                'critic': self.critic.state_dict(),
                'actor': self.policy.state_dict(),
                'vae': self.vae.actor.state_dict()} # self.vae.state_dict()} if not diffusion
            self.logger.export_to_json()
            directory_dict = f'{self.name_save}_mean{self.mean_:.2f}_'\
                f'cvar{self.cvar_:.2f}epoch'\
                f'{self.eval_episode}.tar'
            torch.save(model_dict, directory_dict)
            print('Saving final model. End training')

    @property
    def mean_(self):
        mean = self.mean_eval_cumreward(self.times_eval)
        return mean

    @property
    def cvar_(self):
        cvar = self.cvar_eval_cumreward(self.times_eval)
        return cvar


class Data_Sampler(object):
    def __init__(self, data, device, reward_tune='no'):

        self.state = torch.from_numpy(data['observations']).float()
        self.action = torch.from_numpy(data['actions']).float()
        self.next_state = torch.from_numpy(data['next_observations']).float()
        reward = torch.from_numpy(data['rewards']).view(-1, 1).float()
        self.not_done = 1. - torch.from_numpy(data['terminals']).view(-1, 1).float()

        self.size = self.state.shape[0]
        self.state_dim = self.state.shape[1]
        self.action_dim = self.action.shape[1]

        self.device = device

        if reward_tune == 'normalize':
            reward = (reward - reward.mean()) / reward.std()
        elif reward_tune == 'iql_antmaze':
            reward = reward - 1.0
        elif reward_tune == 'iql_locomotion':
            reward = iql_normalize(reward, self.not_done)
        elif reward_tune == 'cql_antmaze':
            reward = (reward - 0.5) * 4.0
        elif reward_tune == 'antmaze':
            reward = (reward - 0.25) * 2.0
        self.reward = reward

    def sample(self, batch_size):
        ind = torch.randint(0, self.size, size=(batch_size,))

        return (
            self.state[ind].to(self.device),
            self.action[ind].to(self.device),
            self.next_state[ind].to(self.device),
            self.reward[ind].to(self.device),
            self.not_done[ind].to(self.device)
            )


def iql_normalize(reward, not_done):
    trajs_rt = []
    episode_return = 0.0
    for i in range(len(reward)):
        episode_return += reward[i]
        if not not_done[i]:
            trajs_rt.append(episode_return)
            episode_return = 0.0
    rt_max, rt_min = torch.max(torch.tensor(trajs_rt)), torch.min(torch.tensor(trajs_rt))
    reward /= (rt_max - rt_min)
    reward *= 1000.
    return reward


class EarlyStopping_D(object):
    def __init__(self, tolerance=5, min_delta=0):
        self.tolerance = tolerance
        self.min_delta = min_delta
        self.counter = 0
        self.early_stop = False

    def __call__(self, train_loss, validation_loss):
        if (validation_loss - train_loss) > self.min_delta:
            self.counter += 1
            if self.counter >= self.tolerance:
                return True
        else:
            self.counter = 0
        return False