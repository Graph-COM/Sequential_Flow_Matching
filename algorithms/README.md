# algorithms

`algorithms` folder is designed to contain implementation of algorithms or models.
Content in `algorithms` can be loosely grouped components (e.g. models) or an algorithm has already has all
components chained together (e.g. Lightning Module, RL algo).
You should create a folder name after your own algorithm or baselines in it.


The `base_model` implements diffusion, flow matching and meanflow, with API such as `forward()` and `model_sample()`.

The `base_task` implements the training/inference pipelines that will call an agnostic model's API.

The `pipelines` simply combines model with task. For example, `flow_simulation.py` initializes the task `SimulationTask` and let 'SimulationTask.model=FlowMatching()'.
