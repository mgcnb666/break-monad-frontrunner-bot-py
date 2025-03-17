#!/usr/bin/env python3
import logging
import argparse
import toml
import time
import threading
from typing import List, Dict, Tuple
from web3 import Web3

from src.settings.settings import Settings, ApiSettings, GameSettings, EOA
from src.logger.logger import Logs

# Configuration constants class - better organization method
class Config:
    """Configuration constants class, centralized management of all global configurations"""
    # Basic settings
    BALANCE_THRESHOLD = 0.001
    DEFAULT_ATTEMPTS = 10000000
    GAS_LIMIT = 53000
    TARGET_CONTRACT = '0xBce2C725304e09CEf4cD7639760B67f8A0Af5bc4'
    
    # Gas related settings
    GAS_THRESHOLD = 750  # Gas Price threshold (Gwei)
    MAX_GAS_PRICE = 750  # Maximum Gas Price limit (Gwei)
    MONITOR_GAS_PRICE_BONUS = 1  # Additional value for monitored address Gas price
    
    # Monitoring settings
    ENABLE_MONITOR = True  # Control whether to enable wallet monitoring
    BLOCK_LIMIT = 10  # Block quantity limit for monitoring specific addresses
    ANALYSIS_INTERVAL = 10  # Analysis interval (seconds)
    
    # Auto update settings
    AUTO_UPDATE_INTERVAL_MINUTES = 1  # Time interval for auto-updating monitored addresses (minutes)
    AUTO_UPDATE_MIN_TX_COUNT = 10  # Minimum transaction count required for auto-adding monitored addresses
    AUTO_UPDATE_BLOCK_SCAN_LIMIT = 100  # Number of blocks to scan during auto-update

# Global monitored addresses set (using memory storage instead of file)
MONITOR_ADDRESSES: set = set() 

# Thread-safe shared data structure
class SharedData:
    """Provides thread-safe data sharing and state control"""
    def __init__(self):
        self._lock = threading.Lock()
        self._should_run = False
        self._gas_price = 0
        self.analysis_complete = threading.Event()
        self.stop_event = threading.Event()
        
    def update_gas_data(self, should_run: bool, gas_price: float) -> None:
        """Update Gas data, set analysis complete event"""
        with self._lock:
            self._should_run = should_run
            self._gas_price = gas_price
            self.analysis_complete.set()
            
    def get_gas_data(self) -> Tuple[bool, float]:
        """Get Gas data, thread-safe"""
        with self._lock:
            return self._should_run, self._gas_price
    
    def should_stop(self) -> bool:
        """Check if threads should stop"""
        return self.stop_event.is_set()
    
    def signal_stop(self) -> None:
        """Send stop signal to all threads"""
        self.stop_event.set() 

def load_addresses_from_file(file_path: str = None) -> List[str]:
    """Get monitored addresses list from memory set"""
    return list(MONITOR_ADDRESSES)

def wait_with_interrupt(shared_data: SharedData, seconds: float) -> None:
    """Thread-safe wait for specified time, supports early interruption"""
    # Convert seconds to number of 100ms segments
    segments = max(1, int(seconds * 10))
    
    for _ in range(segments):
        if shared_data.should_stop():
            break
        time.sleep(0.1)  # 100 milliseconds 

def get_addresses_gas_prices_optimized(w3: Web3, addresses: List[str], contract_address: str, own_address: str = None) -> Dict[str, float]:
    """
    Get Gas prices for multiple addresses sending transactions to specific contract (batch processing)
    
    Args:
        w3: Web3 object
        addresses: List of addresses
        contract_address: Target contract address
        own_address: Own address (will be excluded)
        
    Returns:
        Dict[str, float]: Mapping from address to its Gas price
    """
    logger = Logs(__name__).log(level=logging.INFO)
    result = {}
    
    # Quick parameter validity check
    if not addresses:
        return result
    
    try:
        # 1. Preprocess addresses - create fast lookup mapping and exclude own address
        own_address_lower = own_address.lower() if own_address else ""
        addresses_map = {addr.lower(): addr for addr in addresses if addr.lower() != own_address_lower}
        
        if not addresses_map:
            return result
        
        # 2. Prepare search state
        contract_address_lower = contract_address.lower()
        addresses_set = set(addresses_map.keys())
        addresses_found = set()
        current_block = w3.eth.block_number
        
        # 3. Efficient block scanning
        for block_number in range(current_block, max(0, current_block - Config.BLOCK_LIMIT), -1):
            # Exit early if all addresses found
            if addresses_found == addresses_set:
                break
                
            try:
                block = w3.eth.get_block(block_number, full_transactions=True)
                
                # Optimize transaction filtering logic
                for tx in block.transactions:
                    # Efficiently filter irrelevant transactions
                    if (not hasattr(tx, 'to') or tx.to is None or not hasattr(tx, 'from') or 
                        hasattr(tx, 'to') and tx.to.lower() != contract_address_lower):
                        continue
                    
                    # Check if sender is a monitoring target
                    tx_from = tx['from'].lower()
                    if tx_from in addresses_set and tx_from not in addresses_found:
                        # Extract Gas price and convert to Gwei
                        gas_price_gwei = float(w3.from_wei(tx.gasPrice, 'gwei'))
                        result[addresses_map[tx_from]] = gas_price_gwei
                        addresses_found.add(tx_from)
            except Exception as e:
                logger.error(f"Error processing block {block_number}: {e}")
        
        # Log summary information
        if addresses_found and len(addresses_found) < len(addresses_set):
            logger.info(f"Found Gas prices for {len(addresses_found)}/{len(addresses_set)} addresses")
    
    except Exception as e:
        logger.error(f"Error getting address Gas prices: {e}")
    
    return result 

def analyze_gas_usage(w3: Web3, contract_address: str, own_address: str) -> Tuple[bool, float]:
    """
    Analyze recent Gas usage and decide whether to run the program
    
    Args:
        w3: Web3 object
        contract_address: Target contract address
        own_address: Own account address
        
    Returns:
        Tuple[bool, float]: (should run, recommended Gas price)
    """
    logger = Logs(__name__).log(level=logging.INFO)
    _enable_monitor = Config.ENABLE_MONITOR  # Local variable to avoid multi-threading issues
    
    try:
        # 1. Initialize and get current network Gas price
        own_address = own_address.lower() if own_address else ""
        contract_address = contract_address.lower() if contract_address else ""
        current_gas_price = float(w3.from_wei(w3.eth.gas_price, 'gwei'))
        
        # 2. Set default values
        base_gas_price = current_gas_price
        final_gas_price = current_gas_price
        highest_gas_price = 0
        
        logger.info(f"Analyzing Gas price - Network: {base_gas_price:.1f} Gwei, Monitoring: {'Enabled' if _enable_monitor else 'Disabled'}")
        
        # 3. If monitoring enabled, get Gas prices of monitored addresses
        if _enable_monitor and (monitor_addresses := load_addresses_from_file()):
            addresses_gas_prices = get_addresses_gas_prices_optimized(
                w3, monitor_addresses, contract_address, own_address)
            
            if addresses_gas_prices:
                # Find highest price
                highest_gas_price = max(addresses_gas_prices.values(), default=0)
                
                # a) If monitored address price is higher, prefer it
                if highest_gas_price > base_gas_price:
                    final_gas_price = highest_gas_price + Config.MONITOR_GAS_PRICE_BONUS
                    logger.info(f"Using monitored address Gas: {highest_gas_price:.1f} + {Config.MONITOR_GAS_PRICE_BONUS} = {final_gas_price:.1f} Gwei")
        
        # 4. Apply maximum Gas price limit
        if final_gas_price > Config.MAX_GAS_PRICE:
            final_gas_price = Config.MAX_GAS_PRICE
            logger.info(f"Gas price exceeds limit, using: {final_gas_price} Gwei")
        
        # 5. Determine running condition
        should_stop = (_enable_monitor and highest_gas_price >= Config.GAS_THRESHOLD) or (final_gas_price >= Config.GAS_THRESHOLD)
        
        # 6. Log decision
        if should_stop:
            if _enable_monitor and highest_gas_price >= Config.GAS_THRESHOLD:
                logger.info(f"Monitored address Gas too high: {highest_gas_price:.1f} >= {Config.GAS_THRESHOLD}, stopping")
            else:
                logger.info(f"Current Gas too high: {final_gas_price:.1f} >= {Config.GAS_THRESHOLD}, stopping")
        
        return not should_stop, final_gas_price
        
    except Exception as e:
        logger.error(f"Gas analysis error: {e}")
        return True, 0  # Default to run when error occurs 

def add_address_to_monitor(address: str) -> bool:
    """
    Add address to monitoring list
    
    Args:
        address: Address to add
        
    Returns:
        bool: Returns True if addition successful
    """
    global MONITOR_ADDRESSES
    
    # Check address format
    if not address or len(address) != 42 or not address.startswith('0x'):
        return False
    
    # Add address to set
    try:
        if address not in MONITOR_ADDRESSES:
            MONITOR_ADDRESSES.add(address)
            Logs(__name__).log().info(f"Added monitoring address: {address}")
        return True
    except Exception:
        return False 

def update_monitor_addresses_from_blocks(w3: Web3, contract_address: str, own_address: str, 
                               block_count: int = None, min_tx_count: int = None) -> None:
    """
    Scan recent blocks for active addresses and update monitoring list
    
    Args:
        w3: Web3 object
        contract_address: Target contract address
        own_address: Own address
        block_count: Number of blocks to scan
        min_tx_count: Minimum transaction count threshold
    """
    global MONITOR_ADDRESSES
    logger = Logs(__name__).log()
    
    # Use default values
    block_count = block_count or Config.AUTO_UPDATE_BLOCK_SCAN_LIMIT
    min_tx_count = min_tx_count or Config.AUTO_UPDATE_MIN_TX_COUNT
    
    # Return directly if monitoring disabled
    if not Config.ENABLE_MONITOR:
        logger.info("Wallet monitoring feature disabled, skipping auto-update")
        return
    
    try:
        # 1. Prepare addresses and block range
        contract_address_lower = contract_address.lower()
        own_address_lower = own_address.lower() if own_address else ""
        current_block = w3.eth.block_number
        start_block = max(0, current_block - block_count)
        
        logger.info(f"Scanning blocks: {start_block} → {current_block} looking for active addresses")
        
        # 2. Efficient block scanning
        address_tx_count = {}
        blocks_scanned = 0
        
        for block_number in range(current_block, start_block, -1):
            try:
                block = w3.eth.get_block(block_number, full_transactions=True)
                blocks_scanned += 1
                
                # Reduce logging, improve efficiency
                if blocks_scanned % 100 == 0:
                    logger.info(f"Scanned: {blocks_scanned} blocks, found {len(address_tx_count)} addresses")
                
                # Process transactions in block
                for tx in block.transactions:
                    # Filter valid transactions
                    if not hasattr(tx, 'to') or tx.to is None or not hasattr(tx, 'from'):
                        continue
                        
                    tx_to = tx.to.lower()
                    if tx_to != contract_address_lower:
                        continue
                    
                    # Process sender address
                    tx_from = tx['from'].lower()
                    if tx_from != own_address_lower:
                        address_tx_count[tx_from] = address_tx_count.get(tx_from, 0) + 1
                        
            except Exception as e:
                logger.error(f"Error processing block {block_number}: {e}")
        
        # 3. Filter active addresses
        active_addresses = []
        for address, count in address_tx_count.items():
            if count >= min_tx_count and len(address) == 42 and address.startswith('0x'):
                active_addresses.append((address, count))
        
        # Sort by transaction count (descending)
        active_addresses.sort(key=lambda x: x[1], reverse=True)
        
        # 4. Update monitoring list
        old_size = len(MONITOR_ADDRESSES)
        MONITOR_ADDRESSES.clear()
        
        # Add new addresses
        added_count = 0
        for address, count in active_addresses:
            try:
                checksum_address = w3.to_checksum_address(address)
                MONITOR_ADDRESSES.add(checksum_address)
                added_count += 1
                
                # Only print first few addresses
                if added_count <= 3:
                    logger.info(f"Added monitoring: {checksum_address[:10]}... (txs:{count})")
            except Exception:
                continue
        
        # 5. Summary log
        if added_count > 3:
            logger.info(f"Added {added_count-3} other active addresses")
        
        logger.info(f"Monitoring list updated: {old_size} → {added_count} addresses")
    
    except Exception as e:
        logger.error(f"Error updating monitoring addresses: {e}") 

def auto_update_monitor_addresses_thread(shared_data, w3, contract_address, own_address,
                               interval_minutes=None, min_tx_count=None, block_count=None) -> None:
    """
    Thread function for auto-updating monitored addresses
    
    Args:
        shared_data: Shared data object
        w3: Web3 object
        contract_address: Target contract address
        own_address: Own address
        interval_minutes: Update interval (minutes)
        min_tx_count: Minimum transaction count requirement
        block_count: Number of blocks to scan
    """
    # Use default values
    interval_minutes = interval_minutes or Config.AUTO_UPDATE_INTERVAL_MINUTES
    min_tx_count = min_tx_count or Config.AUTO_UPDATE_MIN_TX_COUNT
    block_count = block_count or Config.AUTO_UPDATE_BLOCK_SCAN_LIMIT
    
    logger = Logs(__name__).log()
    logger.info(f"Auto-update monitoring address thread started (interval: {interval_minutes} minutes)")
    
    # Convert to seconds
    interval_seconds = interval_minutes * 60
    
    # Execute first scan
    update_monitor_addresses_from_blocks(w3, contract_address, own_address, block_count, min_tx_count)
    
    # Loop until stop signal received
    while not shared_data.should_stop():
        try:
            # Wait for interval time
            wait_with_interrupt(shared_data, interval_seconds)
            
            # Exit loop if stop signal received
            if shared_data.should_stop():
                break
            
            # Execute update
            logger.info(f"Executing periodic monitoring address update")
            update_monitor_addresses_from_blocks(w3, contract_address, own_address, block_count, min_tx_count)
            
        except Exception as e:
            logger.error(f"Auto-update thread error: {e}")
            wait_with_interrupt(shared_data, 10)
    
    logger.info("Auto-update monitoring address thread ended") 

def gas_analysis_thread(shared_data, w3, contract_address, account_address, interval=None) -> None:
    """
    Gas analysis thread, periodically analyzes Gas price
    
    Args:
        shared_data: Shared data object
        w3: Web3 object
        contract_address: Target contract address
        account_address: Account address
        interval: Analysis interval (seconds)
    """
    # Use default value
    interval = interval or Config.ANALYSIS_INTERVAL
    
    logger = Logs(__name__).log()
    logger.info(f"Gas analysis thread started (interval: {interval} seconds)")
    
    # Adaptive interval system
    adaptive_interval = interval
    error_count = 0
    max_error_interval = 30  # Maximum error interval 30 seconds
    
    while not shared_data.should_stop():
        try:
            # 1. Execute Gas analysis
            should_run, gas_price = analyze_gas_usage(w3, contract_address, account_address)
            
            # 2. Update shared data
            shared_data.update_gas_data(should_run, gas_price)
            
            # 3. Reset error count and interval after success
            error_count = 0
            adaptive_interval = interval
            
            # 4. Wait until next analysis cycle
            wait_with_interrupt(shared_data, adaptive_interval)
            
        except Exception as e:
            # Error handling and adaptive interval
            error_count += 1
            logger.error(f"Gas analysis error: {e}")
            
            if error_count > 2:
                # Increase interval after multiple consecutive errors
                adaptive_interval = min(interval * 2, max_error_interval)
                logger.warning(f"Multiple errors, increasing wait time: {adaptive_interval} seconds")
            
            # Brief wait before retry
            wait_with_interrupt(shared_data, 5)
    
    logger.info("Gas analysis thread exited") 

def transaction_thread(shared_data, w3, contract, account, chain_id, gas_limit, args) -> None:
    """
    Transaction sending thread
    
    Args:
        shared_data: Shared data object
        w3: Web3 object
        contract: Contract object
        account: Account object
        chain_id: Chain ID
        gas_limit: Gas limit
        args: Command line arguments
    """
    logger = Logs(__name__).log()
    logger.info("Transaction sending thread started")
    
    # 1. Initialize
    max_attempts = args.attempts if hasattr(args, 'attempts') and args.attempts > 0 else Config.DEFAULT_ATTEMPTS
    tx_interval = args.interval if hasattr(args, 'interval') else 1.0
    skip_gas_check = args.skip_gas_check if hasattr(args, 'skip_gas_check') else False
    tx_count = 0
    error_count = 0
    
    # Get initial nonce
    try:
        nonce = w3.eth.get_transaction_count(account.address)
        logger.info(f"Initial nonce: {nonce}")
    except Exception as e:
        logger.error(f"Failed to get nonce: {e}")
        nonce = 0
    
    # 2. Main transaction loop
    while not shared_data.should_stop() and tx_count < max_attempts:
        try:
            # Wait for Gas analysis to complete
            if not shared_data.analysis_complete.is_set() and not skip_gas_check:
                logger.info("Waiting for Gas analysis to complete...")
                shared_data.analysis_complete.wait(timeout=60)
            
            # Get Gas analysis results
            should_run, gas_price_value = shared_data.get_gas_data()
            
            # Determine whether to send transaction
            if should_run or skip_gas_check:
                # Set Gas price (priority: command line args > analysis result > current network price)
                if hasattr(args, 'gas_price_gwei') and args.gas_price_gwei > 0:
                    gas_price_gwei = args.gas_price_gwei
                elif gas_price_value > 0:
                    gas_price_gwei = int(gas_price_value)
                else:
                    gas_price_gwei = int(w3.from_wei(w3.eth.gas_price, 'gwei'))
                
                # Apply maximum Gas limit
                if gas_price_gwei > Config.MAX_GAS_PRICE:
                    logger.warning(f"Gas price exceeds limit: {gas_price_gwei} → {Config.MAX_GAS_PRICE} Gwei")
                    gas_price_gwei = Config.MAX_GAS_PRICE
                
                # Convert to Wei
                gas_price_wei = w3.to_wei(gas_price_gwei, 'gwei')
                
                # Update nonce
                try:
                    nonce = w3.eth.get_transaction_count(account.address)
                except Exception as e:
                    logger.error(f"Failed to get nonce, using local increment: {e}")
                    nonce += 1
                
                # Send transaction
                try:
                    # Build transaction
                    play_tx = contract.functions.frontrun().build_transaction({
                        'chainId': chain_id,
                        'gas': gas_limit,
                        'gasPrice': gas_price_wei,
                        'nonce': nonce,
                    })
                    
                    # Sign and send
                    signed_tx = account.sign_transaction(play_tx)
                    tx_hash = w3.eth.send_raw_transaction(signed_tx.raw_transaction)
                    tx_hash_hex = tx_hash.hex()
                    
                    logger.info(f"Transaction #{tx_count+1} sent: {tx_hash_hex[:10]}..., Gas: {gas_price_gwei} Gwei")
                    
                    # Update counters and nonce
                    tx_count += 1
                    nonce += 1
                    error_count = 0  # Reset error count
                    
                    # Wait until next transaction cycle
                    if tx_count < max_attempts:
                        logger.info(f"Waiting {tx_interval} seconds before sending next transaction")
                        wait_with_interrupt(shared_data, tx_interval)
                        
                except Exception as e:
                    error_count += 1
                    logger.error(f"Error sending transaction: {e}")
                    
                    # Adjust wait time based on error count
                    wait_time = 5.0 if error_count > 3 else 0.5
                    wait_with_interrupt(shared_data, wait_time)
            else:
                # Not suitable for sending transaction now
                logger.info("Current Gas price not suitable for transaction, waiting")
                wait_with_interrupt(shared_data, 10)
                
        except Exception as e:
            logger.error(f"Transaction thread error: {e}")
            wait_with_interrupt(shared_data, 5)
    
    # 3. End processing
    if tx_count >= max_attempts:
        logger.info(f"Reached maximum attempt count: {max_attempts}")
    
    logger.info("Transaction thread exited") 

def health_check_thread(shared_data, threads):
    """Health check thread, monitors status of other threads"""
    logger = Logs(__name__).log()
    logger.info("Health check thread started")
    
    while not shared_data.should_stop():
        # Check all thread statuses
        for name, thread in threads.items():
            if not thread.is_alive():
                logger.error(f"Thread exception: '{name}' has stopped running")
        
        # Check every 5 seconds
        wait_with_interrupt(shared_data, 5)
    
    logger.info("Health check thread exited")

def monitor_keyboard_thread(shared_data):
    """Keyboard input monitoring thread, for handling user interrupts"""
    logger = Logs(__name__).log()
    logger.info("Keyboard monitoring thread started, press 'q' to exit program")
    
    while not shared_data.should_stop():
        try:
            user_input = input()
            if user_input.lower() == 'q':
                logger.info("Received exit command")
                shared_data.signal_stop()
                break
        except Exception:
            shared_data.signal_stop()
            break
    
    logger.info("Keyboard monitoring thread exited") 

def play_threaded() -> None:
    """Main program entry function"""
    # Get logger
    logger = Logs(__name__).log()
    logger.info("Starting Monad Frontrunner Bot Threaded Version")
    
    # Process command line arguments
    parser = argparse.ArgumentParser(description="Threaded version of Monad Frontrunner Bot.")
    parser.add_argument('--attempts', type=int, default=Config.DEFAULT_ATTEMPTS, help="Number of attempts")
    parser.add_argument('--private_key', type=str, default="", help="Private key")
    parser.add_argument('--gas_price_gwei', type=float, default=0, help="Gas price (Gwei)")
    parser.add_argument('--target_address', type=str, default="", help="Monitoring address")
    parser.add_argument('--max_gas', type=float, default=Config.MAX_GAS_PRICE, help="Maximum Gas price (Gwei)")
    parser.add_argument('--analysis_interval', type=int, default=Config.ANALYSIS_INTERVAL, help="Analysis interval (seconds)")
    parser.add_argument('--interval', type=float, default=1.0, help="Transaction sending interval (seconds)")
    parser.add_argument('--skip_gas_check', action='store_true', help="Skip Gas check")
    parser.add_argument('--disable_monitor', action='store_true', help="Disable wallet monitoring")
    parser.add_argument('--disable_auto_update', action='store_true', help="Disable auto update")
    parser.add_argument('--target_contract', type=str, default=Config.TARGET_CONTRACT, help="Target contract address")
    parser.add_argument('--min_tx_count', type=int, default=Config.AUTO_UPDATE_MIN_TX_COUNT, help="Minimum transaction count")
    parser.add_argument('--block_scan_limit', type=int, default=Config.AUTO_UPDATE_BLOCK_SCAN_LIMIT, help="Block scan limit")
    parser.add_argument('--monitor_gas_bonus', type=int, default=Config.MONITOR_GAS_PRICE_BONUS, help="Monitoring Gas bonus")
    args = parser.parse_args()
    
    try:
        # 1. Load configuration
        try:
            config_file = toml.load('settings.toml')
        except Exception as e:
            logger.error(f"Error loading config file: {e}")
            logger.info("Trying to use example config file...")
            try:
                config_file = toml.load('settings.toml.example')
            except Exception as e2:
                logger.error(f"Loading example config file also failed: {e2}")
                return
        
        # Parse configuration
        settings = Settings(
            api_settings=ApiSettings(**config_file['api_settings']),
            game_settings=GameSettings(**config_file['game_settings']),
            eoa=EOA(**config_file['eoa'])
        )
        
        # 2. Connect to blockchain
        provider_url = settings.api_settings.rpc_url
        contract_abi = settings.game_settings.abi
        logger.info(f"Connecting to node: {provider_url}")
        
        w3 = Web3(Web3.HTTPProvider(provider_url))
        if not w3.is_connected():
            logger.error("Unable to connect to node")
            return
        
        logger.info(f"Successfully connected to node, chain ID: {w3.eth.chain_id}")
        
        # 3. Configure account
        private_key = args.private_key if args.private_key else settings.eoa.private_key
        account = w3.eth.account.from_key(private_key)
        
        # Check balance
        balance = w3.eth.get_balance(account.address)
        balance_eth = w3.from_wei(balance, 'ether')
        logger.info(f"Account: {account.address}, Balance: {balance_eth} ETH")
        
        if balance_eth < Config.BALANCE_THRESHOLD:
            logger.warning(f"Account balance below {Config.BALANCE_THRESHOLD} ETH, may not be able to transact")
        
        # 4. Setup contract
        target_contract = args.target_contract or settings.game_settings.frontrunner_contract_address
        contract = w3.eth.contract(address=target_contract, abi=contract_abi)
        logger.info(f"Target contract: {target_contract}")
        
        # 5. Create shared data and update configuration
        shared_data = SharedData()
        
        # Apply command line arguments to configuration
        Config.MAX_GAS_PRICE = args.max_gas
        Config.TARGET_CONTRACT = target_contract
        Config.AUTO_UPDATE_MIN_TX_COUNT = args.min_tx_count
        Config.AUTO_UPDATE_BLOCK_SCAN_LIMIT = args.block_scan_limit
        Config.MONITOR_GAS_PRICE_BONUS = args.monitor_gas_bonus
        Config.ENABLE_MONITOR = not args.disable_monitor
        
        # 6. Display key configuration
        logger.info(f"Configuration: Max Gas={Config.MAX_GAS_PRICE}Gwei, Monitor bonus={Config.MONITOR_GAS_PRICE_BONUS}")
        logger.info(f"Wallet monitoring status: {'Enabled' if Config.ENABLE_MONITOR else 'Disabled'}")
        
        # Add monitoring address
        if args.target_address:
            add_address_to_monitor(args.target_address)
            logger.info(f"Added target monitoring address: {args.target_address}")
        
        # 7. Initial Gas analysis
        if not args.skip_gas_check:
            try:
                logger.info("Performing initial Gas analysis...")
                should_run, gas_price_value = analyze_gas_usage(w3, target_contract, account.address)
                shared_data.update_gas_data(should_run, gas_price_value)
                logger.info(f"Initial analysis: Run={should_run}, Price={gas_price_value}")
            except Exception as e:
                logger.error(f"Initial Gas analysis failed: {e}")
                default_price = args.gas_price_gwei if args.gas_price_gwei > 0 else 50
                shared_data.update_gas_data(True, default_price)
        else:
            default_price = args.gas_price_gwei if args.gas_price_gwei > 0 else 50
            shared_data.update_gas_data(True, default_price)
            logger.info(f"Skipping Gas analysis, using default price: {default_price} Gwei")
        
        # 8. Create and start threads
        threads = {}
        
        # 8.1 Analysis thread - if not skipping gas check
        if not args.skip_gas_check:
            analysis_thread = threading.Thread(
                target=gas_analysis_thread,
                args=(shared_data, w3, target_contract, account.address, args.analysis_interval),
                daemon=True
            )
            analysis_thread.start()
            threads["analysis"] = analysis_thread
        
        # 8.2 Transaction thread
        tx_thread = threading.Thread(
            target=transaction_thread,
            args=(shared_data, w3, contract, account, w3.eth.chain_id, Config.GAS_LIMIT, args),
            daemon=True
        )
        tx_thread.start()
        threads["transaction"] = tx_thread
        
        # 8.3 Health check thread
        health_thread = threading.Thread(
            target=health_check_thread,
            args=(shared_data, threads),
            daemon=True
        )
        health_thread.start()
        threads["health_check"] = health_thread
        
        # 8.4 Keyboard monitoring thread
        keyboard_thread = threading.Thread(
            target=monitor_keyboard_thread,
            args=(shared_data,),
            daemon=True
        )
        keyboard_thread.start()
        threads["keyboard"] = keyboard_thread
        
        # 8.5 Auto update thread
        if not args.disable_auto_update and Config.ENABLE_MONITOR:
            auto_update_thread = threading.Thread(
                target=auto_update_monitor_addresses_thread,
                args=(shared_data, w3, target_contract, account.address, 
                     Config.AUTO_UPDATE_INTERVAL_MINUTES, args.min_tx_count, args.block_scan_limit),
                daemon=True
            )
            auto_update_thread.start()
            threads["auto_update"] = auto_update_thread
            logger.info("Auto update monitoring address thread started")
        
        logger.info(f"Started {len(threads)} threads, press 'q' key to exit program")
        
        # 9. Main thread wait
        try:
            keyboard_thread.join()
            logger.info("Shutting down all threads...")
            shared_data.signal_stop()
            
            # Wait for threads to end, set timeout to ensure program can exit
            for name, thread in threads.items():
                if name != "keyboard":
                    thread.join(timeout=3)
            
        except KeyboardInterrupt:
            logger.info("Received keyboard interrupt signal")
            shared_data.signal_stop()
        
        logger.info("Program has exited")
        
    except Exception as e:
        logger.error(f"Main program error: {e}")

if __name__ == "__main__":
    play_threaded() 
