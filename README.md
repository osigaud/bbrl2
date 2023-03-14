# `BBRL2`: A Flexible and Simple Library for Reinforcement Learning (RL) Agents

BBRL stands for "BlackBoard Reinforcement Learning". Initially, this library was a fork of [the SaLinA library](https://github.com/facebookresearch/salina). 
But SaLinA is a general model for sequential learning whereas BBRL is dedicated to RL, thus it focuses on a subset of SaLinA. 
Morevover, BBRL is designed for education purpose (in particular, to teach various RL algorithms, concepts and phenomena). 
Thus the fork slowly drifted away from SaLinA and became independent after a few months, even if some parts of the code are still inherited from SaLinA.

## TL;DR :

`BBRL2` is a lightweight library extending PyTorch modules for developing **RL agents**.

It is derived from [`SaLinA`](https://github.com/facebookresearch/salina)  and [`BBRL`](https://github.com/osigaud/bbrl)
* The main difference with `BBRL` and `SaLinA` is that `BBRL2` is compatible with `Gymnasium`:
  * The difference between `NoAutoResetGymAgent` and `AutoResetGymAgent' depends on wether the `Gymnasium` environment contains an `AutoResetWrapper` or not.
  * You should now use `env/stopped` instead of `env/done` as a stop variable
* No multiprocessing / no remote agent or workspace yet (see [ROADMAP.md](ROADMAP.md))
* You can easily save and load your models with `agent.save` and `Agent.load` by making them inherit from `SerializableAgent`, if they are not serializable, you have to override the `serialize` method.
* An ImageGymAgent has been added with adapted serialization
* Many typos have been fixed and type hints have been added

## Citing `BBRL2`
`BBRL2` being inspired from [`SaLinA`](https://github.com/facebookresearch/salina), please use this bibtex if you want to cite `BBRL2` in your publications:


Please use this bibtex if you want to cite this repository in your publications:

Link to the paper: [SaLinA: Sequential Learning of Agents](https://arxiv.org/abs/2110.07910)

```
    @misc{salina,
        author = {Ludovic Denoyer, Alfredo de la Fuente, Song Duong, Jean-Baptiste Gaya, Pierre-Alexandre Kamienny, Daniel H. Thompson},
        title = {SaLinA: Sequential Learning of Agents},
        year = {2021},
        publisher = {Arxiv},salina_cl
        howpublished = {\url{https://gitHub.com/facebookresearch/salina}},
    }
```

## Quick Start

* Just clone the repo and
* with pip 21.3 or newer : `pip install -e .`

**For development, set up [pre-commit](https://pre-commit.com) hooks:**

* Run `pip install pre-commit`
    * or `conda install -c conda-forge pre-commit`
    * or `brew install pre-commit`
* In the top directory of the repo, run `pre-commit install` to set up the git hook scripts
* Now `pre-commit` will run automatically on `git commit`!
* Currently isort, black are used, in that order

## Organization of the repo

* [bbrl2](bbrl2) is the core library
  * [bbrl2.agents](bbrl2/agents) is the catalog of agents (the same as `torch.nn` but for agents)

## Dependencies

`BBRL2` utilizes [`PyTorch`](https://github.com/pytorch/pytorch), [`Hydra`](https://github.com/facebookresearch/hydra) for configuring experiments, and [`Gymnasium`](https://github.com/Farama-Foundation/Gymnasium) for reinforcement learning environments.

## Note on the logger

We provide a simple Logger that logs in both [`TensorBoard`](https://github.com/tensorflow/tensorboard) format and [`wandb`](https://github.com/wandb/wandb), but also as pickle files that can be re-read to make tables and figures. See [logger](bbrl2/logger.py). This logger can be easily replaced by any other logger.

## Description

### What `BBRL2` is

* A sandbox for developing sequential models at scale.

* A small (300 hundred lines) 'core' code that defines everything you will use to implement `agents` involved in sequential decision learning systems.
  * It is easy to understand and use since it keeps the main principles of pytorch, just extending [`nn.Module`](https://pytorch.org/docs/stable/nn.html) to [`Agent`](/bbrl2/agent.py) in order to handle the temporal dimension.
* A set of **agents** that can be combined (like pytorch modules) to obtain complex behaviors
* A set of references implementations and examples in different domains **Reinforcement Learning**, **Imitation Learning**, **Computer Vision**, with more to come...

### What `BBRL2` is not

* A `library`: BBRL2 is just a small layer on top of pytorch that encourages good practices for implementing RL models. Accordingly, it is very simple to understand and use, while very powerful.
* A `framework`: BBRL2 is not a framework, it is just a set of tools that can be used to implement any kind of RL system.

## License

`BBRL2` is released under the MIT license. See [LICENSE](LICENSE) for additional details about it.
