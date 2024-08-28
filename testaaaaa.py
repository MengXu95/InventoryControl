import threading
import time

class TimeoutException(Exception):
    """Custom exception to be raised when a timeout occurs."""
    pass

def timeout_handler():
    """Function that raises a TimeoutException."""
    raise TimeoutException("The function execution exceeded the time limit!")

def run_with_timeout(func, timeout, *args, **kwargs):
    """
    Runs a function with a specified timeout. If the function takes longer than
    `timeout` seconds, it will be stopped, and a TimeoutException will be raised.

    Parameters:
    - func: The function to run.
    - timeout: The maximum time in seconds the function is allowed to run.
    - args, kwargs: Arguments to pass to the function.
    """
    timer = threading.Timer(timeout, timeout_handler)
    timer.start()
    try:
        result = func(*args, **kwargs)
    except TimeoutException as e:
        print(e)
        result = None
    finally:
        timer.cancel()  # Cancel the timer if the function completes within the timeout
    return result

# Example function that takes some time to execute
def long_running_function(seconds):
    print(f"Starting long-running function for {seconds} seconds...")
    time.sleep(seconds)
    print("Function completed.")
    return "Success"

# Example usage
try:
    result = run_with_timeout(long_running_function, 5, 10)  # 5-second timeout, 10-second function
    if result is not None:
        print(f"Function result: {result}")
    else:
        print("Function was terminated due to timeout.")
except TimeoutException as e:
    print(f"Caught an exception: {e}")