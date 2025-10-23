# main.py
import time
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
import torch as T

# ایمپورت آژنت‌ها (مطمئن شو فایل‌ها در مسیر PYTHONPATH قرار دارند یا کنار main.py)
from ddpg_agent import DDPGAgent
from td3_agent import TD3Agent
from sac_agent import SACAgent

SEED = 0

def run_agent(agent, env_name, n_episodes=400, max_steps=1000, seed=SEED):
    env = gym.make(env_name)
    np.random.seed(seed)
    T.manual_seed(seed)
    try:
        env.action_space.seed(seed)
    except Exception:
        pass

    rewards = []
    t0 = time.perf_counter()
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed+ep)
        done = False
        ep_reward = 0.0
        step = 0
        while not done and step < max_steps:
            act = agent.choose_action(obs)  # API مشترک
            # اگر اکشن اسکالر/آرایه است حتما به numpy float تبدیل شود
            action_to_env = np.array(act, dtype=np.float32)
            next_state, reward, terminated, truncated, _ = env.step(action_to_env)
            done = terminated or truncated

            agent.remember(obs, action_to_env, reward, next_state, int(done))
            agent.learn()

            ep_reward += reward
            obs = next_state
            step += 1

        rewards.append(ep_reward)
        avg100 = np.mean(rewards[-100:]) if len(rewards) >= 1 else 0.0
        print(f"[{agent.__class__.__name__}] Ep {ep+1}/{n_episodes} Reward: {ep_reward:.3f} Avg100: {avg100:.3f}")

    t1 = time.perf_counter()
    env.close()
    return rewards, t1 - t0


def moving_average(x, w=20):
    if len(x) < w:
        return np.array(x)
    return np.convolve(x, np.ones(w)/w, mode='valid')


def main():
    ENV_NAME = "MountainCarContinuous-v0"
    N_EPISODES = 400   # برای تست سریع می‌تونی کمتر کنی
    MAX_STEPS = 1000

    # پارامترهای مشترک — مطمئن شو با سازنده‌های کلاس‌ها تطابق داره
    common_params = dict(
        input_dims=[2],
        num_actions=1,
        tau=0.001,
        gamma=0.99,
        max_size=1000000,
        hidden1_dims=400,
        hidden2_dims=300,
        batch_size=128,
        critic_lr=0.001,
        actor_lr=0.0005
    )
    sac_params = {
        **common_params,
        "alpha": 0.01,   # فقط برای SAC
    }
    results = {}
    runtimes = {}

    # DDPG (بدون تغییر داخل کلاس شما)
    print("Running DDPG ...")
    ddpg = DDPGAgent(**common_params)
    rewards_ddpg, t_ddpg = run_agent(ddpg, ENV_NAME, n_episodes=N_EPISODES, max_steps=MAX_STEPS, seed=SEED)
    results['DDPG'] = rewards_ddpg
    runtimes['DDPG'] = t_ddpg
    print(f"DDPG done. Time: {t_ddpg:.2f}s")

    # TD3
    print("Running TD3 ...")
    td3 = TD3Agent(**common_params)   # مطمئن شو امضاها یکسانند
    rewards_td3, t_td3 = run_agent(td3, ENV_NAME, n_episodes=N_EPISODES, max_steps=MAX_STEPS, seed=SEED)
    results['TD3'] = rewards_td3
    runtimes['TD3'] = t_td3
    print(f"TD3 done. Time: {t_td3:.2f}s")

    # SAC
    print("Running SAC ...")
    sac = SACAgent(**sac_params)
    rewards_sac, t_sac = run_agent(sac, ENV_NAME, n_episodes=N_EPISODES, max_steps=MAX_STEPS, seed=SEED)
    results['SAC'] = rewards_sac
    runtimes['SAC'] = t_sac
    print(f"SAC done. Time: {t_sac:.2f}s")

    # رسم نمودار مقایسه پاداش‌ها
    plt.figure(figsize=(10,6))
    for name, scores in results.items():
        plt.plot(scores, label=f"{name} reward")
        ma = moving_average(scores, w=20)
        if len(ma) > 0:
            plt.plot(range(len(ma)), ma, linestyle='--', label=f"{name} MA(20)")
    plt.xlabel("Episode")
    plt.ylabel("Episode reward")
    plt.title(f"Reward per episode — Env: {ENV_NAME}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("rewards_compare.png")
    print("Saved rewards_compare.png")

    # نمودار میله‌ای برای زمان اجرا
    plt.figure(figsize=(6,4))
    names = list(runtimes.keys())
    times = [runtimes[n] for n in names]
    plt.bar(names, times)
    plt.ylabel("Runtime (s)")
    plt.title("Algorithm runtimes")
    plt.tight_layout()
    plt.savefig("runtimes_compare.png")
    print("Saved runtimes_compare.png")

    # نمایش خلاصه
    print("\nSummary:")
    for name, scores in results.items():
        final = scores[-1] if len(scores)>0 else None
        mean100 = np.mean(scores[-100:]) if len(scores)>=1 else None
        print(f" {name}: final_reward={final:.3f} mean_last100={mean100:.3f} runtime={runtimes[name]:.2f}s")

if __name__ == "__main__":
    main()