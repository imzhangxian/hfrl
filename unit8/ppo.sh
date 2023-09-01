python3 ppo.py 
        --env-id="LunarLander-v2" 
        --repo-id="xian79/ppo-LunarLander-v2.8" 
        --ent-coef=0.03 
        --total-timesteps=200000 
        --learning-rate=0.001 
        --num-steps=256

python3 ppo.py \
        --env-id="LunarLander-v2" \
        --ent-coef=0.03 \
        --total-timesteps=500000 \
        --learning-rate=0.01 \
        --num-envs=16 \
        --update-epochs=8 \
        --num-steps=1000 \
        --gae-lambda=0.98 \
        --num-minibatches=32 \
        --gamma=0.999