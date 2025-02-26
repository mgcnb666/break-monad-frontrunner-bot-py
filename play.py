import logging
import argparse
import toml
import time

from src.settings.settings import Settings, ApiSettings, GameSettings, EOA
from src.logger.logger import Logs
from web3 import Web3

BALANCE_THRESHOLD: float = 0.001
DEFAULT_ATTEMPTS: int = 10000000
GAS_LIMIT: int = 200000

def play() -> None:

    parser = argparse.ArgumentParser(description="Break Monad Frontrunner Bot.")
    parser.add_argument('--gas_price_gwei', type=int, default=0, help="Set the gas price in GWEI.")
    parser.add_argument('--attempts', type=int, default=False, help="Number of attempts to play.")
    parser.add_argument('--interval', type=float, default=1, help="Delay between attempts in seconds.")
    args = parser.parse_args()

 
    # Initialize logger
    logging.basicConfig(level=logging.INFO)
    logger = Logs(__name__).log(level=logging.INFO)

    # 1. Load config
    config_file = toml.load('settings.toml')

    # 2. Parse config
    settings = Settings(
        api_settings=ApiSettings(**config_file['api_settings']),
        game_settings=GameSettings(**config_file['game_settings']),
        eoa=EOA(**config_file['eoa'])
    )

    # 3. Initialize web3 client
    w3 = Web3(Web3.HTTPProvider(settings.api_settings.rpc_url))

    # w3
    if not w3.is_connected():
        raise Exception("Failed to connect to the Ethereum network.")
    else:
        logger.info("Connected to the Monad network.")

    # 4. Get frontrunner contract
    contract = w3.eth.contract(
        address=w3.to_checksum_address(settings.game_settings.frontrunner_contract_address),
        abi=settings.game_settings.abi
    )

    DEFAULT_GAS_PRICE: int = int(w3.eth.gas_price*10**-9) if args.gas_price_gwei == 0 else int(args.gas_price_gwei)

    logger.info(f"Using gas price: {DEFAULT_GAS_PRICE} GWEI")

    # 5. Get account
    try:
        account = w3.eth.account.from_key(settings.eoa.private_key)
    except Exception as e:
        logger.error(f"Failed to get account from private key: {e}")
        raise e
    
    logger.info(f"Account to be used: {account.address}")

    # Balance ceck
    balance = w3.from_wei(w3.eth.get_balance(account.address), 'ether')
    logger.info(f"Account balance: {balance} Testnet Monad")

    if balance < BALANCE_THRESHOLD:
        logger.error("Account balance is too low to play. Please add funds to the account.")
        logger.warning("Exiting...")
        time.sleep(1)
        return

    # Score check
    try:
        _ ,wins, losses = contract.functions.getScore(account.address).call()
        if wins > 0 or losses > 0:
            logger.info(f"It looks like it's not the first time: you won {wins} times and lost {losses} times.")
        else:
            logger.info("It looks like it's the first time you play. Good luck!")
    except Exception as e:
        logger.error(f"Failed to get score: {e} - Skipping...")


    nonce: int = w3.eth.get_transaction_count(account.address)
    logger.info(f"Nonce: {nonce}")
    chain_id: int = w3.eth.chain_id

    gas_price_wei: int = w3.to_wei(DEFAULT_GAS_PRICE, 'gwei')

    # if attempts is 0, play 
    if args.attempts == False:
        attempts = DEFAULT_ATTEMPTS
    else:
        attempts = args.attempts

    while True:
        try:
            # Build the transaction with the given nonce and gas price.
            txn = contract.functions.frontrun().build_transaction({
                'chainId': chain_id,
                'gas': GAS_LIMIT,
                'gasPrice': gas_price_wei,
                'nonce': nonce,
            })

            # Sign the transaction with the private key.
            signed_txn = account.sign_transaction(txn)

            # Send the signed transaction.
            tx_hash = w3.eth.send_raw_transaction(signed_txn.raw_transaction)
            logger.info(f"Sent transaction with nonce {nonce}. Tx hash: {tx_hash.hex()}")
        except Exception as e:
            logger.error(f"Error sending transaction with nonce {nonce}: {e}")
    
        nonce += 1
        time.sleep(args.interval)
        attempts -= 1
        if attempts == 0:
            logger.info("Attempts limit reached. Exiting...")
            break
    
    logger.info("All attempts have been made. Exiting...")
    return

if __name__ == "__main__":
    play()