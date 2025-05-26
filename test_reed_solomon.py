from ecc.reed_solomon import ReedSolomonCode
import os

def test_reed_solomon():
    # Initialize the codec
    rs_codec = ReedSolomonCode()
    
    # Test messages including longer ones
    test_messages = [
        "Hello!",  # Short 
        "A" * 223,  # Max length message (k=223)
        os.urandom(200).hex(),  
        "Test"
    ]
    
    error_counts = [1, 2, 3, 10, 15]
    
    for message in test_messages:
        print(f"\nTesting message: {message[:50]}..." if len(message) > 50 else f"\nTesting message: {message}")
        print("-" * 50)
        
        # Convert string to bytes before encoding
        message_bytes = message.encode('utf-8')
        
        # Check message length
        if len(message_bytes) > 223:
            print(f"Warning: Message length {len(message_bytes)} exceeds maximum 223 bytes")
            continue
            
        print(f"Original message length: {len(message_bytes)} bytes")
        encoded = rs_codec.encode(message_bytes)
        print(f"Encoded length: {len(encoded)} bytes")
        
        for num_errors in error_counts:
            print(f"\nTesting with {num_errors} errors:")
            
            # Introduce errors
            corrupted = rs_codec.introduce_errors(encoded, num_errors)
            print(f"Corrupted length: {len(corrupted)} bytes")
            
            try:
                # Try to decode the corrupted message
                decoded = rs_codec.decode(corrupted)
                print(f"Successfully decoded: {decoded[:50]}..." if len(decoded) > 50 else f"Successfully decoded: {decoded}")
                print(f"Matches original: {decoded == message}")
            except ValueError as e:
                print(f"Failed to decode: {str(e)}")

if __name__ == "__main__":
    test_reed_solomon() 