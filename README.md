# MDRL-with-Demonstration-Cloning-for-Target-Localization
This code simulates the MDRL-SR and MDRL-DC models in the paper titled "Multi-Agent Deep Reinforcement Learning with Demonstration Cloning for Target Localization".
The contents of this repository are as follows:
* MDRL-SR: the python files required to train and test an MDRL-SR model for the target localization problem.
* MDRL-DC: the python files required to train and test an MDRL-DC model for the target localization problem.
* Experts: pre-trained experts used in the MDRL-DC model. The folder has 3 experts, previously trained in a single-agent environment with 1, 2, or 3 walls.
* Embeddings: a folder that contains pre-trained encoders (autoencoders) which are used to encode the walls map. Each encoder is pre-trained on datasets of 1, 2, or 3 walls.
* Walls: this folder is to contain 3 walls sets for environment variations of 1, 2, and 3 walls. Each set contains hundreds of thousands of different distributions of walls. This helps in saving time during training instead of creating wall distributions actively at the beginning of each episode. Due to the size limitations in GitHub, the user is required to download the three pkl files to the Walls folder from OneDrive on: https://1drv.ms/u/s!AilzKc-njjP7kN4oc4ku1Q9HxdUtnw?e=F24W8g 


Each of the MDRL-SR and MDRL-DC folders contains the following:
* Main.py: the main file used to train. In the initialization of the Main class, the user needs to set self.NumOfAgents and self.NumOfWalls (max 3 walls). It is recommended to increase self.n_workers for parallelized processing. It is also recommended to train on a machine with GPU (automatically detected).  Throughout the training, the actor policy is stored in a created folder under the name "MultiAgentLocal", which can be later used for testing/inference.
* Environment.py: describes the target localization environment dynamics and reward function.
* PPO1.py: contains the required methods for Proximal Policy Optimization, which is used to create and update the actor/critic networks.
* utils.py: contains a set of utilities used by the other methods.
* test1.py: can be used for inference to test a fully trained model. This simulates the deployment process. Once the model is fully trained, the user can run this file, which creates scenarios of varying environments (different target locations, agents locations, and walls distributions), and reports localization results. The user needs to ensure that NumOfAgents and NumOfWalls are set to the same values in Main.py.
