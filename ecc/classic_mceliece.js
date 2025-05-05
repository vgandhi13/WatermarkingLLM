const { McEliece } = require('mceliece-nist');

class ClassicMcEliece {
    constructor(parameterSet = 'mceliece8192128') {
        this.kem = new McEliece(parameterSet);
        this.parameterSet = parameterSet;
    }

    generateKeys() {
        const keys = this.kem.keypair();
        return {
            publicKey: Buffer.from(keys.publicKey).toString('hex'),
            privateKey: Buffer.from(keys.privateKey).toString('hex')
        };
    }

    encrypt(publicKey) {
        const pubKeyBuffer = Buffer.from(publicKey, 'hex');
        const result = this.kem.generateKey(pubKeyBuffer);
        return {
            key: Buffer.from(result.key).toString('hex'),
            encryptedKey: Buffer.from(result.encryptedKey).toString('hex')
        };
    }

    decrypt(encryptedKey, privateKey) {
        const encKeyBuffer = Buffer.from(encryptedKey, 'hex');
        const privKeyBuffer = Buffer.from(privateKey, 'hex');
        const result = this.kem.decryptKey(privKeyBuffer, encKeyBuffer);
        return Buffer.from(result).toString('hex');
    }
}

// Command-line interface
if (require.main === module) {
    const mc = new ClassicMcEliece();
    const command = process.argv[2];
    
    try {
        switch (command) {
            case 'generate':
                // Immediately generate and output keys
                const keys = mc.generateKeys();
                console.log(JSON.stringify(keys));
                break;
                
            case 'encrypt':
                // Handle encryption through stdin
                process.stdin.setEncoding('utf8');
                let encryptInput = '';
                process.stdin.on('data', chunk => encryptInput += chunk);
                process.stdin.on('end', () => {
                    const data = JSON.parse(encryptInput);
                    const result = mc.encrypt(data.publicKey);
                    console.log(JSON.stringify(result));
                });
                break;
                
            case 'decrypt':
                // Handle decryption through stdin
                process.stdin.setEncoding('utf8');
                let decryptInput = '';
                process.stdin.on('data', chunk => decryptInput += chunk);
                process.stdin.on('end', () => {
                    const data = JSON.parse(decryptInput);
                    const decryptedKey = mc.decrypt(data.encryptedKey, data.privateKey);
                    console.log(JSON.stringify({ decryptedKey }));
                });
                break;
                
            default:
                console.error('Unknown command');
                process.exit(1);
        }
    } catch (error) {
        console.error(error.message);
        process.exit(1);
    }
}

module.exports = ClassicMcEliece;