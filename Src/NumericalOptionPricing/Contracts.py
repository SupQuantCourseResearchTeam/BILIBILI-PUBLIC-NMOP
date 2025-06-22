from dataclasses import dataclass
from typing import Dict, Union
import json


@dataclass
class MonteCarloControlParams:
    n_sample_paths: int
    use_control_variate: bool


@dataclass
class PDEControlParams:
    n_steps_S: int
    n_steps_T: int


@dataclass
class VanillaContract:
    contract_type: str
    s0: float
    T: float
    K: float
    r: float
    sigma: float
    algorithm_type: str
    control_params: Union[None, MonteCarloControlParams, PDEControlParams]


def load_contract(json_path: str) -> VanillaContract:
    # Check the json contract path
    if not json_path.endswith(".json"):
        raise ValueError(f"Arg 'json_path' must end with '.json'.")

    # Read the json contract
    with open(json_path, "r") as file:
        contract_json = json.load(file)

    # Build the control params
    control_params = None
    if contract_json["algorithm_type"] == "MonteCarlo":
        use_control_variate = True
        if contract_json["control_params"]["use_control_variate"] == "false":
            use_control_variate = False
        control_params = MonteCarloControlParams(
            n_sample_paths=contract_json["control_params"]["n_sample_paths"],
            use_control_variate=use_control_variate,
        )
    if contract_json["algorithm_type"] == "PDE":
        control_params = PDEControlParams(
            n_steps_S=contract_json["control_params"]["n_steps_S"],
            n_steps_T=contract_json["control_params"]["n_steps_T"],
        )

    # Parse the json contract into a vanilla contract object
    return VanillaContract(
        contract_type=contract_json["contract_type"],
        s0=contract_json["spot"],
        T=contract_json["maturity"],
        K=contract_json["strike"],
        r=contract_json["ir"],
        sigma=contract_json["vol"],
        algorithm_type=contract_json["algorithm_type"],
        control_params=control_params,
    )
