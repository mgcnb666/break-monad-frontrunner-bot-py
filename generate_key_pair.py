from web3 import Account

if __name__ == "__main__":
    # Generate a new random account

    account = Account.create()

    # Get the private key in hexadecimal format
    private_key = account.key.hex()

    # Get the address associated with the private key
    address = account.address

    # Output the private key and address
    print(f"Private Key: {private_key} !DONT SHARE THIS!")
    print(f"Public Address: {address} <-- send Testnet DMON to this address to play")

