import logging
from datetime import datetime
from pathlib import Path
from solib.datatypes import Question
from solib.utils import str_config
from solib.llm_utils import parallelized_call
from solib.protocols.protocols import *
from solib.protocols.judges import *
from solib.protocols.abstract import QA_Agent, Judge, Protocol

LOGGER = logging.getLogger(__name__)


class Experiment:
    """Experiment parameterization.

    To run an experiment, call `experiment()`.

    Properties:
        agent_models: list[str]
        agent_toolss: list[list[callable]]
        judge_models: list[str]
        protocols: dict[str, type[Protocol]]
        num_turnss: list[int]
        agents: list[QA_Agent]
        judges: list[Judge]
        other_componentss: dict[str, list[dict[str, Any]]]
        init_kwargss: dict[str, list[dict[str, Any]]]
        all_configs: list[dict[str, Any]]
        filtered_configs: list[dict[str, Any]]

    Methods:
        experiment: Run the experiment.
        filter_config: Filter the configurations. Subclass this to decide which configurations
            to run.
        get_path: Get the path to write the results to.
    """

    def __init__(
        self,
        questions: list[Question],
        agent_models: list[str],
        agent_toolss: list[list[callable]],
        judge_models: list[str],
        protocols: dict[str, type[Protocol]] = None,
        num_turnss: list[int] = None,
        write_path: Path = Path("experiments")
        / f"results_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}",
    ):
        """
        Args:
            questions: Data to experiment on.
            agent_models: List of models for the agents.
            agent_toolss: List of tools for the agents.
            judge_models: List of models for the judges.
            protocols: Dictionary of protocols to run.
            num_turnss: List of number of turns for the protocols.
            write_path: Folder directory to write the results to.
        """
        self.questions = questions
        self.agent_models = agent_models
        self.agent_toolss = agent_toolss
        self.judge_models = judge_models
        if protocols is None:
            pass
        elif isinstance(protocols, list):
            self.protocols = {k: v for k, v in self.protocols.items() if k in protocols}
        elif isinstance(protocols, dict):
            self.protocols = protocols
        else:
            raise ValueError(f"protocols must be a list or dict, got {type(protocols)}")
        if num_turnss is None:
            num_turnss = [2, 4]
        self.num_turnss = num_turnss
        self.write_path = write_path

    protocols = {
        "blind": Blind,
        "propaganda": Propaganda,
        "debate": Debate,
        "consultancy": Consultancy,
    }

    @property
    def agents(self):
        return [
            QA_Agent(
                model=model,
                tools=tools,
            )
            for model in self.agent_models
            for tools in self.agent_toolss
        ]

    @property
    def judges(self):
        tot_judges = [
            TipOfTongueJudge(model) for model in self.judge_models if model != "human"
        ]
        jap_judges = [JustAskProbabilityJudge(model) for model in self.judge_models]
        return tot_judges + jap_judges

    @property
    def other_componentss(self):
        return {
            "blind": [{}],
            "propaganda": [{}],
            "debate": [{"adversary": agent} for agent in self.agents],
            "consultancy": [{}],
        }

    @property
    def init_kwargss(self):
        init_kwargss_debate = [
            {"simultaneous": t, "num_turns": n}
            for t in [True, False]
            for n in self.num_turnss
        ]
        init_kwargss_consultancy = [
            {"consultant_goes_first": t, "num_turns": n}
            for t in [True, False]
            for n in self.num_turnss
        ]
        return {
            "blind": [{}],
            "propaganda": [{}],
            "debate": init_kwargss_debate,
            "consultancy": init_kwargss_consultancy,
        }

    @property
    def all_configs(self):
        return [
            {
                "protocol": protocol,
                "init_kwargs": init_kwargs,
                "call_kwargs": {
                    "agent": agent,
                    "judge": judge,
                    **other_components,
                },
            }
            for protocol_name, protocol in self.protocols.items()
            for init_kwargs in self.init_kwargss[protocol_name]
            for agent in self.agents
            for judge in self.judges
            for other_components in self.other_componentss[protocol_name]
        ]

    def filter_config(self, config: dict):
        """Subclass this. By default, uses _filter_selfplay"""
        return self._filter_selfplay(config)

    @property
    def filtered_configs(self):
        return [config for config in self.all_configs if self.filter_config(config)]

    async def experiment(self):

        async def run_experiment(config: dict):
            setup = config["protocol"](**config["init_kwargs"])
            await setup.experiment(
                questions=self.questions,
                **config["call_kwargs"],
                write=self.get_path(config),
            )

        confirm = input(
            f"Run {len(self.filtered_configs)} experiments? (y/N) [l to list]"
        )
        if confirm.lower() == "l":
            print(str_config(self.filtered_configs))
            confirm = input("Continue? (y/N)")
        if confirm.lower() != "y":
            return
        LOGGER.debug(self.filtered_configs)
        await parallelized_call(run_experiment, self.filtered_configs)

    def _filter_trivial(self, config: dict):
        return True

    def _filter_selfplay(self, config: dict):
        if config["protocol"] == "debate":
            return config["call_kwargs"]["adversary"] != config["call_kwargs"]["agent"]
        return True

    def _filter_nohf(self, config: dict):
        for component in config["call_kwargs"].values():
            if isinstance(component, (QA_Agent, Judge)):
                if component.model.startswith("hf:"):
                    return False
        return True

    def get_path(self, config: dict):
        init_kwargs_str = ""
        for k, v in config["init_kwargs"].items():
            if k == "num_turns":
                k_ = "n"
                v_ = v
            elif k in ["simultaneous", "consultant_goes_first"]:
                k_ = "t"
                v_ = int(v)
            else:
                k_ = k
                v_ = v
            v_ = str(v_)
            init_kwargs_str += f"{k_}{v_}_"
        init_kwargs_str = init_kwargs_str[:-1]
        call_kwargs_str = ""
        for k, v in config["call_kwargs"].items():
            if k in ["agent", "adversary"]:
                k_ = "A"
                v_ = v.model
            elif k == "judge":
                k_ = "J"
                v_ = v.model
            else:
                k_ = k
                v_ = v
            call_kwargs_str += f"{k_}_{v_}_"
        call_kwargs_str = call_kwargs_str[:-1]
        path = (
            self.write_path
            / (config["protocol"].__name__ + "_" + init_kwargs_str)
            / call_kwargs_str
        )
        # path.mkdir(parents=True, exist_ok=True)
        i = 0
        path_new = path
        while path_new.exists():
            i += 1
            path_new = path.with_name(path.stem + f"_{i}")
        return path_new
