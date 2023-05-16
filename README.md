# EBDAC
Risk-Sensitive Offline Reinforcement Learning with Generalizable Behavior Policy Modelling

## Installation

 EBDAC build upon O-RAAC which required the following setup:

```
pip install -e .
```

In order to run EBDAC you need to install D4RL (follow the instructions in https://github.com/rail-berkeley/d4rl) and  you need MuJoCo as a dependency. You may need to obtain a license and follow the setup instructions for mujoco_py. This mostly involves copying the key to your MuJoCo installation folder.


## Training 
To train the GBDAC model you need to:
Activate the environment:
`source venv/bin/activate`
Run the code:
`python3 experiments/gbdac.py --config_name 'name_json_file'`

where 'name_json_file' is a .json file stored in json_params/GBDAC.
We provide the json files with the default parameters we used for each 
environment to get the results in the paper.
To optimize for different risk distortions modify the field in the json_file
or provide it as an argument accordingly.
--RISK_DISTORTION 'risk_distortion'
where 'risk_distortion' can be 'cvar' or 'cpw' or 'wang'.
 
## Evaluation:
The best trained models (according to 0.1 CVaR and Mean metrics) for each environment are saved in the `model-zoo` folder.
To evaluate the policies using such models you can do:
`python3 experiments/ebdac.py --model_path 'name of environment'`
where 'name of environment' can be:
* halfcheetah-medium
* halfcheetah-expert
* walker2d-medium
* walker2d-expert
* hopper-medium
* hopper-expert



