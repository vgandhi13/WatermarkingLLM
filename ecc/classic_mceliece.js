const { McEliece } = require('mceliece-nist');

class ClassicMcEliece {
    constructor(parameterSet = 'mceliece8192128') {
        this.kem = new McEliece(parameterSet);
        this.parameterSet = parameterSet;
        this.publicKey = null;
        this.privateKey = null;
    }

    generateKeys() {
        const keys = this.kem.keypair();
        this.publicKey = keys.publicKey;
        this.privateKey = keys.privateKey;
        return keys;
    }

    //this is basically the encryption function
    generateKey(publicKey = null) {
        if (!publicKey) {
            publicKey = this.publicKey;
        }
        return this.kem.generateKey(publicKey);
    }

    decryptKey(encryptedKey, privateKey = null) {
        if (!privateKey) {
            privateKey = this.privateKey;
        }
        try {
            return this.kem.decryptKey(privateKey, encryptedKey);
        } catch (error) {
            throw new Error(`Decryption failed: ${error.message}`);
        }
    }
}


function testClassicMcEliece() {

    const mc = new ClassicMcEliece();
    
    try {

        const keys = mc.generateKeys();
        console.log('Keys generated successfully');
        console.log(`Using parameter set: ${mc.parameterSet}`);

        const { key, encryptedKey } = mc.generateKey();
        console.log('\nKey generation successful');
        console.log(`Original key: ${key.toString('hex')}`);
        console.log(`Encrypted key length: ${encryptedKey.length} bytes`);
        

        const decryptedKey = mc.decryptKey(encryptedKey);
        console.log('\nDecryption successful');
        console.log(`Decrypted key: ${decryptedKey.toString('hex')}`);
        console.log(`Keys match: ${key.equals(decryptedKey)}`);
        
    } catch (error) {
        console.error(`Error: ${error.message}`);
    }
}


if (require.main === module) {
    testClassicMcEliece();
}

module.exports = ClassicMcEliece;