from phe import paillier

def public_encryption(plaintext, public_key):
    ciphertext = public_key.encrypt(plaintext)
    return ciphertext

def private_decrypt(ciphertext, private_key):
    plaintext = private_key.decrypt(ciphertext)
    return plaintext

def generate_key_pair():
    public_key, private_key = paillier.generate_paillier_keypair()
    return (public_key, private_key)


