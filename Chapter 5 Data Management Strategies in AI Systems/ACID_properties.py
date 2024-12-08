from contextlib import contextmanager
from threading import Lock

class DataStore:
    """Simple demonstration of transaction failures and rollbacks"""
    
    def __init__(self):
        self._data = {}
        self._temp = {}
        self.lock = Lock()
    
    @contextmanager
    def transaction(self):
        try:
            with self.lock:
                print("Starting transaction...")
                print(f"Before transaction - Main data: {self._data}")
                
                self._temp = self._data.copy()
                print(f"Created temporary copy: {self._temp}")
                
                yield self._temp
                
                print(f"Transaction completed - Temporary data: {self._temp}")
                self._data = self._temp
                print(f"Changes committed - Main data: {self._data}")
                
        except Exception as e:
            print(f"Transaction failed! Rolling back...")
            print(f"Error: {str(e)}")
            print(f"Main data (unchanged): {self._data}")
            self._temp = {}
            raise

def demonstrate_failures():
    store = DataStore()
    
    print("\n1. Successful Transaction")
    print("=" * 50)
    try:
        with store.transaction() as data:
            data['balance'] = 1000
            print(f"Operation: Setting balance to 1000")
    except Exception as e:
        print(f"Exception caught: {e}")
    
    print("\n2. Failed Transaction - Division by Zero")
    print("=" * 50)
    try:
        with store.transaction() as data:
            data['balance'] = 2000
            print(f"Operation: Setting balance to 2000")
            result = data['balance'] / 0  # This will fail
            print("This line won't execute")
    except Exception as e:
        print(f"Exception caught: {e}")
    
    print("\n3. Failed Transaction - Invalid Operation")
    print("=" * 50)
    try:
        with store.transaction() as data:
            data['balance'] += 500  # Will fail if 'balance' doesn't exist
            print("This line won't execute")
    except Exception as e:
        print(f"Exception caught: {e}")
    
    print("\n4. Final State Check")
    print("=" * 50)
    print(f"Final state of data store: {store._data}")

if __name__ == "__main__":
    demonstrate_failures()