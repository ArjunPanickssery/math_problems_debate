import logging
from datetime import datetime
from pathlib import Path

from solib.data.loading import Dataset
from solib.utils import str_config, write_json, dump_config, random
from solib.utils import parallelized_call
from solib.utils.llm_utils import SIMULATE, is_local
from solib.protocols.protocols import (
    Blind,
    Propaganda,
    Debate,
    Consultancy,
)
from solib.protocols.judges import (
    TipOfTongueJudge,
    JustAskProbabilityJudge,
    JustAskProbabilitiesJudge,
)
from solib.protocols.agents import BestOfN_Agent
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
        questions: Dataset,
        judge_models: list[str],
        agent_models: list[str],
        agent_toolss: list[list[callable]] = None,
        protocols: dict[str, type[Protocol]] = None,
        num_turnss: list[int] = None,
        bon_ns: list[int] = None,
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
        self.default_quant_config = True
        self.questions = questions
        self.agent_models = agent_models
        self.agent_toolss = agent_toolss if agent_toolss is not None else [[]]
        self.judge_models = judge_models

        if SIMULATE:
            LOGGER.debug("Running in simulation mode, skipping local models...")
            self.agent_models = [
                model for model in self.agent_models if not is_local(model)
            ]
            self.judge_models = [
                model for model in self.judge_models if not is_local(model)
            ]

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
        if bon_ns is None:
            bon_ns = []
        self.num_turnss = num_turnss
        self.bon_ns = bon_ns
        self.write_path = write_path

    protocols = {
        "blind": Blind,
        "propaganda": Propaganda,
        "debate": Debate,
        "consultancy": Consultancy,
    }

    @property
    def agents_plain(self):
        return [
            QA_Agent(
                model=model,
                tools=tools,
            )
            for model in self.agent_models
            for tools in self.agent_toolss
        ]

    @property
    def agents_bestofn(self):
        return [
            BestOfN_Agent(n=n, agent=agent)
            for n in self.bon_ns
            for agent in self.agents_plain
        ]

    @property
    def agents(self):
        return self.agents_plain + self.agents_bestofn

    @property
    def judges(self):
        tot_judges = [
            TipOfTongueJudge(model) for model in self.judge_models if model != "human"
        ]
        jap_judges = [JustAskProbabilityJudge(model) for model in self.judge_models]
        japs_judges = [JustAskProbabilitiesJudge(model) for model in self.judge_models]
        return tot_judges + jap_judges + japs_judges

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
        """Subclass this. By default, uses _filter_selfplay and _filter_nolocaljap and _filter_noapitot."""
        return (
            self._filter_selfplay(config)
            and self._filter_nolocaljap(config)
            and self._filter_noapitot(config)
            and self._filter_nojaps(config)
        )

    @property
    def filtered_configs(self):
        return [config for config in self.all_configs if self.filter_config(config)]

    async def experiment(self, max_configs=None):
        filtered_configs = self.filtered_configs
        random(filtered_configs).shuffle(filtered_configs)
        filtered_configs = filtered_configs[:max_configs]

        async def run_experiment(config: dict):
            LOGGER.status(f"Running experiment {self.get_path(config)}")
            setup = config["protocol"](**config["init_kwargs"])
            stuff = await setup.experiment(
                questions=self.questions,
                **config["call_kwargs"],
                write=self.get_path(config),
            )
            results, stats = stuff
            return stats

        confirm = input(
            f"Run {len(filtered_configs)} experiments on "
            f"{len(self.questions)} questions? (y/N) [l to list]"
        )
        if confirm.lower() == "l":
            print(str_config(filtered_configs))
            confirm = input("Continue? (y/N)")
        if confirm.lower() != "y":
            raise Exception("Experiment aborted by user.")
        LOGGER.debug(filtered_configs)
        statss = await parallelized_call(
            run_experiment, filtered_configs, use_tqdm=True, max_concurrent_queries=5
        )
        all_stats = [
            {"config": config, "stats": stats}
            for config, stats in zip(filtered_configs, statss)
        ]
        write_json(dump_config(all_stats), path=self.write_path / "all_stats.json")

    def _filter_trivial(self, config: dict):
        return True

    def _filter_nojap(self, config: dict):
        for component in config["call_kwargs"].values():
            if isinstance(component, JustAskProbabilityJudge) and not isinstance(
                component, JustAskProbabilitiesJudge
            ):
                return False
        return True

    def _filter_nojaps(self, config: dict):
        for component in config["call_kwargs"].values():
            if isinstance(component, JustAskProbabilitiesJudge):
                return False
        return True

    def _filter_selfplay(self, config: dict):
        if config["protocol"] == "debate":
            return config["call_kwargs"]["adversary"] != config["call_kwargs"]["agent"]
        return True

    def _filter_nolocal(self, config: dict):
        for component in config["call_kwargs"].values():
            if isinstance(component, (QA_Agent, Judge)):
                if is_local(component.model):
                    return False
        return True

    def _filter_nolocaljap(self, config: dict):
        for component in config["call_kwargs"].values():
            if isinstance(
                component, (JustAskProbabilitiesJudge, JustAskProbabilityJudge)
            ) and is_local(component.model):
                return False
        return True

    def _filter_noapitot(self, config: dict):  # avoid doing ToT judge for API models
        for component in config["call_kwargs"].values():
            if isinstance(component, (TipOfTongueJudge)) and not is_local(
                component.model
            ):
                return False
        return True

    def _get_path_protocol(self, config: dict):
        protocol_str = config["protocol"]
        if not isinstance(protocol_str, str):
            protocol_str = protocol_str.__name__
        # we support config["protocol"]: str too because we also use this
        # in reading from all_stats.json
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
            init_kwargs_str += f"_{k_}{v_}"
        return config["protocol"].__name__ + init_kwargs_str

    def _get_path_call(self, config: dict):
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
            call_kwargs_str += f"_{k_}{v_}"
        return call_kwargs_str

    def get_path(self, config: dict):
        path = (
            self.write_path
            / self._get_path_protocol(config)
            / self._get_path_call(config)
        )
        # path.mkdir(parents=True, exist_ok=True)
        i = 0
        path_new = path
        while path_new.exists():
            i += 1
            path_new = path.with_name(path.stem + f"_{i}")
        return path_new
