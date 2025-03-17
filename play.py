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

# 配置常量类 - 更好的组织方式
class Config:
    """配置常量类，集中管理所有全局配置"""
    # 基础设置
    BALANCE_THRESHOLD = 0.001
    DEFAULT_ATTEMPTS = 10000000
    GAS_LIMIT = 56000
    TARGET_CONTRACT = '0xBce2C725304e09CEf4cD7639760B67f8A0Af5bc4'
    
    # Gas相关设置
    GAS_THRESHOLD = 150  # Gas Price阈值（Gwei）
    MAX_GAS_PRICE = 150  # 最高Gas Price限制（Gwei）
    MONITOR_GAS_PRICE_BONUS = 1  # 监控地址Gas价格的额外增加值
    
    # 监控设置
    ENABLE_MONITOR = True  # 控制是否启用监控钱包功能
    BLOCK_LIMIT = 50  # 监控特定地址的区块数量限制
    ANALYSIS_INTERVAL = 10  # 分析间隔（秒）
    
    # 自动更新设置
    AUTO_UPDATE_INTERVAL_MINUTES = 5  # 自动更新监控地址的时间间隔（分钟）
    AUTO_UPDATE_MIN_TX_COUNT = 20  # 自动添加监控地址所需的最小交易次数
    AUTO_UPDATE_BLOCK_SCAN_LIMIT = 500  # 自动更新时扫描的区块数量

# 全局监控地址集合（使用内存存储替代文件）
MONITOR_ADDRESSES: set = set() 

# 线程安全的共享数据结构
class SharedData:
    """提供线程安全的数据共享和状态控制"""
    def __init__(self):
        self._lock = threading.Lock()
        self._should_run = False
        self._gas_price = 0
        self.analysis_complete = threading.Event()
        self.stop_event = threading.Event()
        
    def update_gas_data(self, should_run: bool, gas_price: float) -> None:
        """更新Gas数据，设置分析完成事件"""
        with self._lock:
            self._should_run = should_run
            self._gas_price = gas_price
            self.analysis_complete.set()
            
    def get_gas_data(self) -> Tuple[bool, float]:
        """获取Gas数据，线程安全"""
        with self._lock:
            return self._should_run, self._gas_price
    
    def should_stop(self) -> bool:
        """检查是否应该停止线程"""
        return self.stop_event.is_set()
    
    def signal_stop(self) -> None:
        """发送停止信号给所有线程"""
        self.stop_event.set() 

def load_addresses_from_file(file_path: str = None) -> List[str]:
    """从内存集合中获取监控地址列表"""
    return list(MONITOR_ADDRESSES)

def wait_with_interrupt(shared_data: SharedData, seconds: float) -> None:
    """线程安全地等待指定时间，支持提前中断"""
    # 将秒数转换为100毫秒的片段数量
    segments = max(1, int(seconds * 10))
    
    for _ in range(segments):
        if shared_data.should_stop():
            break
        time.sleep(0.1)  # 100毫秒 

def get_addresses_gas_prices_optimized(w3: Web3, addresses: List[str], contract_address: str, own_address: str = None) -> Dict[str, float]:
    """
    获取多个地址向特定合约发送交易的Gas价格（批量处理）
    
    Args:
        w3: Web3对象
        addresses: 地址列表
        contract_address: 目标合约地址
        own_address: 自己的地址（将被排除）
        
    Returns:
        Dict[str, float]: 地址到其Gas价格的映射
    """
    logger = Logs(__name__).log(level=logging.INFO)
    result = {}
    
    # 快速检查参数有效性
    if not addresses:
        return result
    
    try:
        # 1. 预处理地址 - 创建快速查找映射并排除自己的地址
        own_address_lower = own_address.lower() if own_address else ""
        addresses_map = {addr.lower(): addr for addr in addresses if addr.lower() != own_address_lower}
        
        if not addresses_map:
            return result
        
        # 2. 准备搜索状态
        contract_address_lower = contract_address.lower()
        addresses_set = set(addresses_map.keys())
        addresses_found = set()
        current_block = w3.eth.block_number
        
        # 3. 高效扫描区块
        for block_number in range(current_block, max(0, current_block - Config.BLOCK_LIMIT), -1):
            # 如果找到所有地址则提前退出
            if addresses_found == addresses_set:
                break
                
            try:
                block = w3.eth.get_block(block_number, full_transactions=True)
                
                # 优化交易过滤逻辑
                for tx in block.transactions:
                    # 高效过滤无关交易
                    if (not hasattr(tx, 'to') or tx.to is None or not hasattr(tx, 'from') or 
                        hasattr(tx, 'to') and tx.to.lower() != contract_address_lower):
                        continue
                    
                    # 检查发送者是否是监控目标
                    tx_from = tx['from'].lower()
                    if tx_from in addresses_set and tx_from not in addresses_found:
                        # 提取Gas价格并转换为Gwei
                        gas_price_gwei = float(w3.from_wei(tx.gasPrice, 'gwei'))
                        result[addresses_map[tx_from]] = gas_price_gwei
                        addresses_found.add(tx_from)
            except Exception as e:
                logger.error(f"处理区块 {block_number} 时出错: {e}")
        
        # 日志概要信息
        if addresses_found and len(addresses_found) < len(addresses_set):
            logger.info(f"找到 {len(addresses_found)}/{len(addresses_set)} 个地址的Gas价格")
    
    except Exception as e:
        logger.error(f"获取地址Gas价格时出错: {e}")
    
    return result 

def analyze_gas_usage(w3: Web3, contract_address: str, own_address: str) -> Tuple[bool, float]:
    """
    分析最近交易的Gas使用情况并决定是否运行程序
    
    Args:
        w3: Web3对象
        contract_address: 目标合约地址
        own_address: 自己的账户地址
        
    Returns:
        Tuple[bool, float]: (是否应运行, 建议的Gas价格)
    """
    logger = Logs(__name__).log(level=logging.INFO)
    _enable_monitor = Config.ENABLE_MONITOR  # 局部变量避免多线程问题
    
    try:
        # 1. 初始化并获取当前网络Gas价格
        own_address = own_address.lower() if own_address else ""
        contract_address = contract_address.lower() if contract_address else ""
        current_gas_price = float(w3.from_wei(w3.eth.gas_price, 'gwei'))
        
        # 2. 设置默认值
        base_gas_price = current_gas_price
        final_gas_price = current_gas_price
        highest_gas_price = 0
        
        logger.info(f"分析Gas价格 - 网络: {base_gas_price:.1f} Gwei, 监控: {'启用' if _enable_monitor else '禁用'}")
        
        # 3. 如果启用监控，获取监控地址的Gas价格
        if _enable_monitor and (monitor_addresses := load_addresses_from_file()):
            addresses_gas_prices = get_addresses_gas_prices_optimized(
                w3, monitor_addresses, contract_address, own_address)
            
            if addresses_gas_prices:
                # 找出最高价格
                highest_gas_price = max(addresses_gas_prices.values(), default=0)
                
                # a) 如果监控地址价格更高，优先使用
                if highest_gas_price > base_gas_price:
                    final_gas_price = highest_gas_price + Config.MONITOR_GAS_PRICE_BONUS
                    logger.info(f"使用监控地址Gas: {highest_gas_price:.1f} + {Config.MONITOR_GAS_PRICE_BONUS} = {final_gas_price:.1f} Gwei")
        
        # 4. 应用最高Gas价格限制
        if final_gas_price > Config.MAX_GAS_PRICE:
            final_gas_price = Config.MAX_GAS_PRICE
            logger.info(f"Gas价格超过限制，使用: {final_gas_price} Gwei")
        
        # 5. 判断运行条件
        should_stop = (_enable_monitor and highest_gas_price >= Config.GAS_THRESHOLD) or (final_gas_price >= Config.GAS_THRESHOLD)
        
        # 6. 记录决策
        if should_stop:
            if _enable_monitor and highest_gas_price >= Config.GAS_THRESHOLD:
                logger.info(f"监控地址Gas过高: {highest_gas_price:.1f} >= {Config.GAS_THRESHOLD}，停止")
            else:
                logger.info(f"当前Gas过高: {final_gas_price:.1f} >= {Config.GAS_THRESHOLD}，停止")
        
        return not should_stop, final_gas_price
        
    except Exception as e:
        logger.error(f"Gas分析出错: {e}")
        return True, 0  # 出错时默认运行 

def add_address_to_monitor(address: str) -> bool:
    """
    向监控列表添加地址
    
    Args:
        address: 要添加的地址
        
    Returns:
        bool: 添加成功返回True
    """
    global MONITOR_ADDRESSES
    
    # 检查地址格式
    if not address or len(address) != 42 or not address.startswith('0x'):
        return False
    
    # 添加地址到集合
    try:
        if address not in MONITOR_ADDRESSES:
            MONITOR_ADDRESSES.add(address)
            Logs(__name__).log().info(f"添加监控地址: {address}")
        return True
    except Exception:
        return False 

def update_monitor_addresses_from_blocks(w3: Web3, contract_address: str, own_address: str, 
                               block_count: int = None, min_tx_count: int = None) -> None:
    """
    扫描最近区块寻找活跃地址并更新监控列表
    
    Args:
        w3: Web3对象
        contract_address: 目标合约地址
        own_address: 自己的地址
        block_count: 扫描区块数
        min_tx_count: 最小交易次数阈值
    """
    global MONITOR_ADDRESSES
    logger = Logs(__name__).log()
    
    # 使用默认值
    block_count = block_count or Config.AUTO_UPDATE_BLOCK_SCAN_LIMIT
    min_tx_count = min_tx_count or Config.AUTO_UPDATE_MIN_TX_COUNT
    
    # 监控禁用则直接返回
    if not Config.ENABLE_MONITOR:
        logger.info("监控钱包功能已禁用，跳过自动更新")
        return
    
    try:
        # 1. 准备地址和块范围
        contract_address_lower = contract_address.lower()
        own_address_lower = own_address.lower() if own_address else ""
        current_block = w3.eth.block_number
        start_block = max(0, current_block - block_count)
        
        logger.info(f"扫描区块: {start_block} → {current_block} 寻找活跃地址")
        
        # 2. 高效扫描区块
        address_tx_count = {}
        blocks_scanned = 0
        
        for block_number in range(current_block, start_block, -1):
            try:
                block = w3.eth.get_block(block_number, full_transactions=True)
                blocks_scanned += 1
                
                # 减少日志，提高效率
                if blocks_scanned % 100 == 0:
                    logger.info(f"已扫描: {blocks_scanned} 个区块，找到 {len(address_tx_count)} 个地址")
                
                # 处理区块中的交易
                for tx in block.transactions:
                    # 过滤有效交易
                    if not hasattr(tx, 'to') or tx.to is None or not hasattr(tx, 'from'):
                        continue
                        
                    tx_to = tx.to.lower()
                    if tx_to != contract_address_lower:
                        continue
                    
                    # 处理发送者地址
                    tx_from = tx['from'].lower()
                    if tx_from != own_address_lower:
                        address_tx_count[tx_from] = address_tx_count.get(tx_from, 0) + 1
                        
            except Exception as e:
                logger.error(f"处理区块 {block_number} 出错: {e}")
        
        # 3. 筛选活跃地址
        active_addresses = []
        for address, count in address_tx_count.items():
            if count >= min_tx_count and len(address) == 42 and address.startswith('0x'):
                active_addresses.append((address, count))
        
        # 按交易次数排序（降序）
        active_addresses.sort(key=lambda x: x[1], reverse=True)
        
        # 4. 更新监控列表
        old_size = len(MONITOR_ADDRESSES)
        MONITOR_ADDRESSES.clear()
        
        # 添加新地址
        added_count = 0
        for address, count in active_addresses:
            try:
                checksum_address = w3.to_checksum_address(address)
                MONITOR_ADDRESSES.add(checksum_address)
                added_count += 1
                
                # 只打印前几个地址
                if added_count <= 3:
                    logger.info(f"添加监控: {checksum_address[:10]}... (交易:{count}次)")
            except Exception:
                continue
        
        # 5. 总结日志
        if added_count > 3:
            logger.info(f"添加了 {added_count-3} 个其他活跃地址")
        
        logger.info(f"监控列表更新: {old_size} → {added_count} 个地址")
    
    except Exception as e:
        logger.error(f"更新监控地址出错: {e}") 

def auto_update_monitor_addresses_thread(shared_data, w3, contract_address, own_address,
                               interval_minutes=None, min_tx_count=None, block_count=None) -> None:
    """
    自动更新监控地址的线程函数
    
    Args:
        shared_data: 共享数据对象
        w3: Web3对象
        contract_address: 目标合约地址
        own_address: 自己的地址
        interval_minutes: 更新间隔(分钟)
        min_tx_count: 最小交易次数要求
        block_count: 扫描区块数量
    """
    # 使用默认值
    interval_minutes = interval_minutes or Config.AUTO_UPDATE_INTERVAL_MINUTES
    min_tx_count = min_tx_count or Config.AUTO_UPDATE_MIN_TX_COUNT
    block_count = block_count or Config.AUTO_UPDATE_BLOCK_SCAN_LIMIT
    
    logger = Logs(__name__).log()
    logger.info(f"自动更新监控地址线程启动 (间隔: {interval_minutes}分钟)")
    
    # 转换为秒
    interval_seconds = interval_minutes * 60
    
    # 执行首次扫描
    update_monitor_addresses_from_blocks(w3, contract_address, own_address, block_count, min_tx_count)
    
    # 循环直到收到停止信号
    while not shared_data.should_stop():
        try:
            # 等待间隔时间
            wait_with_interrupt(shared_data, interval_seconds)
            
            # 如果收到停止信号，退出循环
            if shared_data.should_stop():
                break
            
            # 执行更新
            logger.info(f"执行定期更新监控地址")
            update_monitor_addresses_from_blocks(w3, contract_address, own_address, block_count, min_tx_count)
            
        except Exception as e:
            logger.error(f"自动更新线程错误: {e}")
            wait_with_interrupt(shared_data, 10)
    
    logger.info("自动更新监控地址线程结束") 

def gas_analysis_thread(shared_data, w3, contract_address, account_address, interval=None) -> None:
    """
    Gas分析线程，周期性分析Gas价格
    
    Args:
        shared_data: 共享数据对象
        w3: Web3对象
        contract_address: 目标合约地址
        account_address: 账户地址
        interval: 分析间隔(秒)
    """
    # 使用默认值
    interval = interval or Config.ANALYSIS_INTERVAL
    
    logger = Logs(__name__).log()
    logger.info(f"Gas分析线程启动 (间隔: {interval}秒)")
    
    # 自适应间隔系统
    adaptive_interval = interval
    error_count = 0
    max_error_interval = 30  # 最大错误间隔30秒
    
    while not shared_data.should_stop():
        try:
            # 1. 执行Gas分析
            should_run, gas_price = analyze_gas_usage(w3, contract_address, account_address)
            
            # 2. 更新共享数据
            shared_data.update_gas_data(should_run, gas_price)
            
            # 3. 成功后重置错误计数和间隔
            error_count = 0
            adaptive_interval = interval
            
            # 4. 等待到下一分析周期
            wait_with_interrupt(shared_data, adaptive_interval)
            
        except Exception as e:
            # 错误处理和自适应间隔
            error_count += 1
            logger.error(f"Gas分析错误: {e}")
            
            if error_count > 2:
                # 连续多次错误后延长间隔
                adaptive_interval = min(interval * 2, max_error_interval)
                logger.warning(f"多次错误，增加等待时间: {adaptive_interval}秒")
            
            # 短暂等待后重试
            wait_with_interrupt(shared_data, 5)
    
    logger.info("Gas分析线程退出") 

def transaction_thread(shared_data, w3, contract, account, chain_id, gas_limit, args) -> None:
    """
    交易发送线程
    
    Args:
        shared_data: 共享数据对象
        w3: Web3对象
        contract: 合约对象
        account: 账户对象
        chain_id: 链ID
        gas_limit: Gas限制
        args: 命令行参数
    """
    logger = Logs(__name__).log()
    logger.info("交易发送线程启动")
    
    # 1. 初始化
    max_attempts = args.attempts if hasattr(args, 'attempts') and args.attempts > 0 else Config.DEFAULT_ATTEMPTS
    tx_interval = args.interval if hasattr(args, 'interval') else 1.0
    skip_gas_check = args.skip_gas_check if hasattr(args, 'skip_gas_check') else False
    tx_count = 0
    error_count = 0
    
    # 获取初始nonce
    try:
        nonce = w3.eth.get_transaction_count(account.address)
        logger.info(f"初始nonce: {nonce}")
    except Exception as e:
        logger.error(f"获取nonce失败: {e}")
        nonce = 0
    
    # 2. 主交易循环
    while not shared_data.should_stop() and tx_count < max_attempts:
        try:
            # 等待Gas分析完成
            if not shared_data.analysis_complete.is_set() and not skip_gas_check:
                logger.info("等待Gas分析完成...")
                shared_data.analysis_complete.wait(timeout=60)
            
            # 获取Gas分析结果
            should_run, gas_price_value = shared_data.get_gas_data()
            
            # 判断是否应该发送交易
            if should_run or skip_gas_check:
                # 设置Gas价格 (优先级: 命令行参数 > 分析结果 > 网络当前价格)
                if hasattr(args, 'gas_price_gwei') and args.gas_price_gwei > 0:
                    gas_price_gwei = args.gas_price_gwei
                elif gas_price_value > 0:
                    gas_price_gwei = int(gas_price_value)
                else:
                    gas_price_gwei = int(w3.from_wei(w3.eth.gas_price, 'gwei'))
                
                # 应用最高Gas限制
                if gas_price_gwei > Config.MAX_GAS_PRICE:
                    logger.warning(f"Gas价格超限: {gas_price_gwei} → {Config.MAX_GAS_PRICE} Gwei")
                    gas_price_gwei = Config.MAX_GAS_PRICE
                
                # 转换为Wei
                gas_price_wei = w3.to_wei(gas_price_gwei, 'gwei')
                
                # 更新nonce
                try:
                    nonce = w3.eth.get_transaction_count(account.address)
                except Exception as e:
                    logger.error(f"获取nonce失败，使用本地递增: {e}")
                    nonce += 1
                
                # 发送交易
                try:
                    # 构建交易
                    play_tx = contract.functions.frontrun().build_transaction({
                        'chainId': chain_id,
                        'gas': gas_limit,
                        'gasPrice': gas_price_wei,
                        'nonce': nonce,
                    })
                    
                    # 签名并发送
                    signed_tx = account.sign_transaction(play_tx)
                    tx_hash = w3.eth.send_raw_transaction(signed_tx.raw_transaction)
                    tx_hash_hex = tx_hash.hex()
                    
                    logger.info(f"交易 #{tx_count+1} 发送: {tx_hash_hex[:10]}..., Gas: {gas_price_gwei} Gwei")
                    
                    # 更新计数器和nonce
                    tx_count += 1
                    nonce += 1
                    error_count = 0  # 重置错误计数
                    
                    # 等待到下一交易周期
                    if tx_count < max_attempts:
                        logger.info(f"等待 {tx_interval} 秒后发送下一笔交易")
                        wait_with_interrupt(shared_data, tx_interval)
                        
                except Exception as e:
                    error_count += 1
                    logger.error(f"发送交易错误: {e}")
                    
                    # 根据错误次数调整等待时间
                    wait_time = 5.0 if error_count > 3 else 0.5
                    wait_with_interrupt(shared_data, wait_time)
            else:
                # 当前不适合发送交易
                logger.info("当前Gas价格不适合交易，等待")
                wait_with_interrupt(shared_data, 10)
                
        except Exception as e:
            logger.error(f"交易线程错误: {e}")
            wait_with_interrupt(shared_data, 5)
    
    # 3. 结束处理
    if tx_count >= max_attempts:
        logger.info(f"已达到最大尝试次数: {max_attempts}")
    
    logger.info("交易线程退出") 

def health_check_thread(shared_data, threads):
    """健康检查线程，监控其他线程的状态"""
    logger = Logs(__name__).log()
    logger.info("健康检查线程启动")
    
    while not shared_data.should_stop():
        # 检查所有线程状态
        for name, thread in threads.items():
            if not thread.is_alive():
                logger.error(f"线程异常: '{name}' 已停止运行")
        
        # 5秒检查一次
        wait_with_interrupt(shared_data, 5)
    
    logger.info("健康检查线程退出")

def monitor_keyboard_thread(shared_data):
    """监控键盘输入线程，用于处理用户中断"""
    logger = Logs(__name__).log()
    logger.info("键盘监控线程启动，按 'q' 退出程序")
    
    while not shared_data.should_stop():
        try:
            user_input = input()
            if user_input.lower() == 'q':
                logger.info("收到退出命令")
                shared_data.signal_stop()
                break
        except Exception:
            shared_data.signal_stop()
            break
    
    logger.info("键盘监控线程退出") 

def play_threaded() -> None:
    """主程序入口函数"""
    # 获取日志记录器
    logger = Logs(__name__).log()
    logger.info("启动Monad Frontrunner Bot多线程版本")
    
    # 处理命令行参数
    parser = argparse.ArgumentParser(description="多线程版Monad Frontrunner Bot.")
    parser.add_argument('--attempts', type=int, default=Config.DEFAULT_ATTEMPTS, help="尝试次数")
    parser.add_argument('--private_key', type=str, default="", help="私钥")
    parser.add_argument('--gas_price_gwei', type=float, default=0, help="Gas价格(Gwei)")
    parser.add_argument('--target_address', type=str, default="", help="监控地址")
    parser.add_argument('--max_gas', type=float, default=Config.MAX_GAS_PRICE, help="最高Gas价格(Gwei)")
    parser.add_argument('--analysis_interval', type=int, default=Config.ANALYSIS_INTERVAL, help="分析间隔(秒)")
    parser.add_argument('--interval', type=float, default=1.0, help="交易发送间隔(秒)")
    parser.add_argument('--skip_gas_check', action='store_true', help="跳过Gas检查")
    parser.add_argument('--disable_monitor', action='store_true', help="禁用钱包监控")
    parser.add_argument('--disable_auto_update', action='store_true', help="禁用自动更新")
    parser.add_argument('--target_contract', type=str, default=Config.TARGET_CONTRACT, help="目标合约地址")
    parser.add_argument('--min_tx_count', type=int, default=Config.AUTO_UPDATE_MIN_TX_COUNT, help="最小交易次数")
    parser.add_argument('--block_scan_limit', type=int, default=Config.AUTO_UPDATE_BLOCK_SCAN_LIMIT, help="扫描区块数")
    parser.add_argument('--monitor_gas_bonus', type=int, default=Config.MONITOR_GAS_PRICE_BONUS, help="监控Gas加成")
    args = parser.parse_args()
    
    try:
        # 1. 加载配置
        try:
            config_file = toml.load('settings.toml')
        except Exception as e:
            logger.error(f"加载配置文件出错: {e}")
            logger.info("尝试使用示例配置文件...")
            try:
                config_file = toml.load('settings.toml.example')
            except Exception as e2:
                logger.error(f"加载示例配置文件也失败: {e2}")
                return
        
        # 解析配置
        settings = Settings(
            api_settings=ApiSettings(**config_file['api_settings']),
            game_settings=GameSettings(**config_file['game_settings']),
            eoa=EOA(**config_file['eoa'])
        )
        
        # 2. 连接区块链
        provider_url = settings.api_settings.rpc_url
        contract_abi = settings.game_settings.abi
        logger.info(f"连接节点: {provider_url}")
        
        w3 = Web3(Web3.HTTPProvider(provider_url))
        if not w3.is_connected():
            logger.error("无法连接到节点")
            return
        
        logger.info(f"成功连接到节点，链ID: {w3.eth.chain_id}")
        
        # 3. 配置账户
        private_key = args.private_key if args.private_key else settings.eoa.private_key
        account = w3.eth.account.from_key(private_key)
        
        # 检查余额
        balance = w3.eth.get_balance(account.address)
        balance_eth = w3.from_wei(balance, 'ether')
        logger.info(f"账户: {account.address}, 余额: {balance_eth} ETH")
        
        if balance_eth < Config.BALANCE_THRESHOLD:
            logger.warning(f"账户余额低于 {Config.BALANCE_THRESHOLD} ETH，可能无法交易")
        
        # 4. 设置合约
        target_contract = args.target_contract or settings.game_settings.frontrunner_contract_address
        contract = w3.eth.contract(address=target_contract, abi=contract_abi)
        logger.info(f"目标合约: {target_contract}")
        
        # 5. 创建共享数据和更新配置
        shared_data = SharedData()
        
        # 应用命令行参数到配置
        Config.MAX_GAS_PRICE = args.max_gas
        Config.TARGET_CONTRACT = target_contract
        Config.AUTO_UPDATE_MIN_TX_COUNT = args.min_tx_count
        Config.AUTO_UPDATE_BLOCK_SCAN_LIMIT = args.block_scan_limit
        Config.MONITOR_GAS_PRICE_BONUS = args.monitor_gas_bonus
        Config.ENABLE_MONITOR = not args.disable_monitor
        
        # 6. 显示关键配置
        logger.info(f"配置: 最高Gas={Config.MAX_GAS_PRICE}Gwei, 监控加成={Config.MONITOR_GAS_PRICE_BONUS}")
        logger.info(f"监控钱包状态: {'启用' if Config.ENABLE_MONITOR else '禁用'}")
        
        # 添加监控地址
        if args.target_address:
            add_address_to_monitor(args.target_address)
            logger.info(f"已添加目标监控地址: {args.target_address}")
        
        # 7. 初始Gas分析
        if not args.skip_gas_check:
            try:
                logger.info("执行初始Gas分析...")
                should_run, gas_price_value = analyze_gas_usage(w3, target_contract, account.address)
                shared_data.update_gas_data(should_run, gas_price_value)
                logger.info(f"初始分析: 运行={should_run}, 价格={gas_price_value}")
            except Exception as e:
                logger.error(f"初始Gas分析失败: {e}")
                default_price = args.gas_price_gwei if args.gas_price_gwei > 0 else 50
                shared_data.update_gas_data(True, default_price)
        else:
            default_price = args.gas_price_gwei if args.gas_price_gwei > 0 else 50
            shared_data.update_gas_data(True, default_price)
            logger.info(f"跳过Gas分析，使用默认价格: {default_price} Gwei")
        
        # 8. 创建并启动线程
        threads = {}
        
        # 8.1 分析线程 - 如果不跳过gas检查
        if not args.skip_gas_check:
            analysis_thread = threading.Thread(
                target=gas_analysis_thread,
                args=(shared_data, w3, target_contract, account.address, args.analysis_interval),
                daemon=True
            )
            analysis_thread.start()
            threads["analysis"] = analysis_thread
        
        # 8.2 交易线程
        tx_thread = threading.Thread(
            target=transaction_thread,
            args=(shared_data, w3, contract, account, w3.eth.chain_id, Config.GAS_LIMIT, args),
            daemon=True
        )
        tx_thread.start()
        threads["transaction"] = tx_thread
        
        # 8.3 健康检查线程
        health_thread = threading.Thread(
            target=health_check_thread,
            args=(shared_data, threads),
            daemon=True
        )
        health_thread.start()
        threads["health_check"] = health_thread
        
        # 8.4 键盘监控线程
        keyboard_thread = threading.Thread(
            target=monitor_keyboard_thread,
            args=(shared_data,),
            daemon=True
        )
        keyboard_thread.start()
        threads["keyboard"] = keyboard_thread
        
        # 8.5 自动更新线程
        if not args.disable_auto_update and Config.ENABLE_MONITOR:
            auto_update_thread = threading.Thread(
                target=auto_update_monitor_addresses_thread,
                args=(shared_data, w3, target_contract, account.address, 
                     Config.AUTO_UPDATE_INTERVAL_MINUTES, args.min_tx_count, args.block_scan_limit),
                daemon=True
            )
            auto_update_thread.start()
            threads["auto_update"] = auto_update_thread
            logger.info("自动更新监控地址线程已启动")
        
        logger.info(f"已启动 {len(threads)} 个线程，按 'q' 键退出程序")
        
        # 9. 主线程等待
        try:
            keyboard_thread.join()
            logger.info("关闭所有线程...")
            shared_data.signal_stop()
            
            # 等待线程结束，设置超时保证程序能够退出
            for name, thread in threads.items():
                if name != "keyboard":
                    thread.join(timeout=3)
            
        except KeyboardInterrupt:
            logger.info("接收到键盘中断信号")
            shared_data.signal_stop()
        
        logger.info("程序已退出")
        
    except Exception as e:
        logger.error(f"主程序发生错误: {e}")

if __name__ == "__main__":
    play_threaded() 
