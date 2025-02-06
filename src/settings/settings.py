from dataclasses import dataclass
import json

@dataclass
class ApiSettings:
    rpc_url: str

@dataclass
class GameSettings:
    frontrunner_contract_address: str
    abi_string: str

    def __post_init__(self):
        self.abi = self._parse_abi_string(self.abi_string)
    
    def _parse_abi_string(self, abi_string):
        """
        Parses a JSON-formatted ABI string into a Python object.

        :param abi_string: str, a JSON string representing the contract's ABI.
        :return: The parsed ABI (typically a list of dictionaries).
        :raises ValueError: If the ABI string is not valid JSON.
        """
        try:
            abi = json.loads(abi_string)
        except json.JSONDecodeError as e:
            raise ValueError("Failed to decode ABI string. Please ensure it is valid JSON.") from e
        return abi

@dataclass
class EOA:
    private_key: str


@dataclass
class Settings:
    api_settings: ApiSettings
    game_settings: GameSettings
    eoa: EOA


