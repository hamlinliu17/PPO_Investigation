# PPO_Investigation
Final Project for ECE 239AS (Reinforcement learning) with Professor Lin Yang. We did a basic investigation on proximal policy optimization (PPO) and tried training different network architectures around it. 


### Files:

* `PPO.py`: The main python source file that contains the source code for the
    `PPO` object that can train and update the actor and critic network according
    to Proximal Policy Optimization

* `notebooks/`: Notebooks folder (Here is a subset of the notebooks that we used. All of them can be found in this drive folder link: https://drive.google.com/drive/folders/1VaV-ge_SGCTJeEYOTqgypeuNGFk-Vguv?usp=sharing)
    * `terri-search_PPO_main_notebook_MLP1.ipynb`: notebook containing the training loop to train the fully connected neural net architecture on the LunarLander environment
    * `terri_ANT_TANH_PPO_main_notebook.ipynb`: notebook containing the training loop to train the fully connected neural net architecture on the RoboschoolAnt environment
    * `Steven_ANT_ReLU_PPO_main_notebook.ipynb`: notebook containing the training loop to train the fully connected neural net architecture using ReLU as activation on the RoboschoolAnt environment
    * `PPO_main_notebook.ipynb`: notebook containing the main training loop
    * `hamlin_PPO_main_notebook.ipynb`: notebook containing the training loop to train the convolution neural net architecture using ReLU as activation on the RoboschoolAnt environment


### To Run:

We ran all of the notebooks in Google's Colab environment. The links of which can be found here:
https://drive.google.com/drive/folders/1VaV-ge_SGCTJeEYOTqgypeuNGFk-Vguv?usp=sharing


### Results:

![](ant.gif)
Snapshot of one of the test episodes with the Roboschool ant environment

![](lunarlander.gif)

Snapshot of one of the test episodes with the LunarLander environment


### Acknowledgements and References:

We would like to acknowledge Nikhil Barhate and his code repository [here](https://github.com/nikhilbarhate99/PPO-PyTorch) for providing us a starting point and base code for implementing PPO.
